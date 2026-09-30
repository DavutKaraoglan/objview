#!/usr/bin/env python3
"""Turntable OBJ/GLB showcase drawn with quadrant-block pixels in the terminal."""
import io, json, math, os, select, shutil, struct, sys, termios, time, tty
import numpy as np
from PIL import Image

PALETTE = [(226, 74, 74), (86, 200, 110), (78, 126, 235), (235, 196, 78), (196, 92, 206), (86, 206, 214)]
DIST = 3.5
PITCH = 0.4
YAW = 0.0
ZOOM = 1.0  # scales the projection instead of the eye distance, so closing in never warps
PANX = PANY = 0.0
LIGHT = (-0.2796, 0.4660, 0.8388)
GLTYPE = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
GLSIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
# One glyph per subset of the cell's four quadrants, bits TL/TR/BL/BR.
GLYPH = " \u2598\u259d\u2580\u2596\u258c\u259e\u259b\u2597\u259a\u2590\u259c\u2584\u2599\u259f\u2588"
RAMP = " .:-=+abo*#%@"  # `text` mode: one character per cell, density instead of blocks


def palette(groups):
    # A palette per group reads as parts on a real model but as confetti once every
    # triangle is its own group, as in an OBJ without object splits.
    n = groups[-1] + 1
    return [PALETTE[g % len(PALETTE)] for g in groups] if n <= 64 else [PALETTE[3]] * len(groups)


def load_obj(path):
    verts, faces, groups, g = [], [], [], 0
    for ln in open(path):
        f = ln.split()
        if f[:1] == ["v"]:
            verts.append(f[1:4])
        elif f[:1] == ["f"]:
            p = [int(t.split("/")[0]) - 1 for t in f[1:]]
            faces += [(p[0], p[k], p[k + 1]) for k in range(1, len(p) - 1)]
            groups += [g] * (len(p) - 2)
            g += 1
    return np.array(verts, float), np.array(faces), palette(groups)


def load_glb(path):
    raw = open(path, "rb").read()
    js, blob, off = None, None, 12
    while off < len(raw):
        size, kind = struct.unpack_from("<II", raw, off)
        body = raw[off + 8: off + 8 + size]
        js = json.loads(body) if kind == 0x4E4F534A else js
        blob = body if kind == 0x004E4942 else blob
        off += 8 + size

    def accessor(idx):
        a = js["accessors"][idx]
        bv = js["bufferViews"][a["bufferView"]]
        dt = np.dtype(GLTYPE[a["componentType"]])
        n = GLSIZE[a["type"]]
        stride = bv.get("byteStride") or n * dt.itemsize
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        rows = np.frombuffer(blob, "u1", a["count"] * stride, start).reshape(a["count"], stride)
        return np.ascontiguousarray(rows[:, : n * dt.itemsize]).view(dt).reshape(a["count"], n)

    def placement(node):
        m = np.eye(4)
        if "matrix" in node:
            return np.array(node["matrix"], float).reshape(4, 4).T
        if "scale" in node:
            m[:3, :3] = np.diag(node["scale"])
        if "rotation" in node:
            x, y, z, r = node["rotation"]
            m[:3, :3] = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * r), 2 * (x * z + y * r)],
                [2 * (x * y + z * r), 1 - 2 * (x * x + z * z), 2 * (y * z - x * r)],
                [2 * (x * z - y * r), 2 * (y * z + x * r), 1 - 2 * (x * x + y * y)]]) @ m[:3, :3]
        if "translation" in node:
            m[:3, 3] = node["translation"]
        return m

    cache = {}

    def texel(prim, idx):
        mat = js.get("materials", [])[prim["material"]] if "material" in prim else {}
        pbr = mat.get("pbrMetallicRoughness", {})
        tint = np.array(pbr.get("baseColorFactor", [1, 1, 1]), float)[:3] * 255
        tex = pbr.get("baseColorTexture")
        uvk = "TEXCOORD_%d" % tex.get("texCoord", 0) if tex else None
        if uvk not in prim["attributes"]:
            return np.tile(tint, (len(idx), 1))
        si = js["textures"][tex["index"]]["source"]
        if si not in cache:
            bv = js["bufferViews"][js["images"][si]["bufferView"]]
            o = bv.get("byteOffset", 0)
            im = Image.open(io.BytesIO(blob[o: o + bv["byteLength"]]))
            # One texel per triangle, so full resolution is wasted -- but an atlas
            # shrunk too far bleeds unrelated islands into each other.
            im.thumbnail((1024, 1024))
            cache[si] = np.asarray(im.convert("RGB"), float)
        img = cache[si]
        # A triangle is at most a pixel or two on screen, so its centre texel is enough.
        uv = accessor(prim["attributes"][uvk]).astype(float)[idx].mean(1) % 1.0
        px = (uv * (img.shape[1] - 1, img.shape[0] - 1)).astype(int)
        return img[px[:, 1], px[:, 0]] * tint / 255

    verts, faces, colors, base = [], [], [], 0

    def walk(ni, parent):
        nonlocal base
        node = js["nodes"][ni]
        m = parent @ placement(node)
        for prim in js["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []:
            p = accessor(prim["attributes"]["POSITION"]).astype(float) @ m[:3, :3].T + m[:3, 3]
            idx = accessor(prim["indices"]).reshape(-1, 3)
            verts.append(p)
            faces.append(idx + base)
            colors.append(texel(prim, idx))
            base += len(p)
        for child in node.get("children", []):
            walk(child, m)

    for n in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(n, np.eye(4))
    return np.vstack(verts), np.vstack(faces), np.vstack(colors)


TEXT = "text" in sys.argv[1:]
CHARS = RAMP if TEXT else GLYPH
args = [a for a in sys.argv[1:] if a != "text"]
src = args[0] if args else "models/cube.obj"
for cand in (src, f"{sys.path[0]}/{src}", f"{sys.path[0]}/models/{src}"):
    if os.path.exists(cand):
        src = cand
        break
verts, TI, rgbs = load_glb(src) if src.lower().endswith(".glb") else load_obj(src)

V0 = verts - verts.mean(0)
V0 /= np.linalg.norm(V0, axis=1).max()
TI0, COL0 = TI, np.array(rgbs, float)
# The turntable sweeps x and z through each other, so the widest the model ever
# gets is its radius in that plane; height is independent.
RX = np.hypot(V0[:, 0], V0[:, 2]).max()
RY = np.abs(V0[:, 1]).max()
HALF = np.float32(0.5)


def detail(n):
    """Weld vertices onto an n-cell grid. A mesh carries far more detail than a
    terminal shows, and dropping the excess is what keeps the frame loop at 30 fps."""
    global V, TI, COL, NRM, PLANE, UNIT
    key = np.floor((V0 + 1) * (n / 2)).astype(np.int64).clip(0, n - 1)
    _, first, inv = np.unique(key @ (n * n, n, 1), return_index=True, return_inverse=True)
    t = inv[TI0]
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    V, TI, COL = V0[first], t[keep], COL0[keep]
    # Face normals live in model space, so a frame only has to dot them with the
    # unrotated eye and light instead of rebuilding a cross product per triangle.
    A = V[TI[:, 0]]
    NRM = np.cross(V[TI[:, 1]] - A, V[TI[:, 2]] - A)
    PLANE = (NRM * A).sum(1)
    UNIT = NRM / np.linalg.norm(NRM, axis=1, keepdims=True).clip(1e-12)
    # Single precision is plenty for a few thousand pixels and halves the memory
    # traffic, which is what a per-frame pass over 60k triangles is actually bound by.
    V, NRM, PLANE, UNIT, COL = (a.astype(np.float32) for a in (V, NRM, PLANE, UNIT, COL))


def unrotate(x, y, z, ca, sa, cp, sp):
    zp = z * cp - y * sp
    return np.array([x * ca - zp * sa, y * cp + z * sp, x * sa + zp * ca], np.float32)


def frame(ang):
    cols, rows = shutil.get_terminal_size()
    w, h = cols * 2, (rows - 1) * 2
    ca, sa, cp, sp = math.cos(ang), math.sin(ang), math.cos(PITCH), math.sin(PITCH)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    X = x * ca + z * sa
    zp = z * ca - x * sa
    Y = y * cp - zp * sp
    Z = y * sp + zp * cp - DIST
    s = min(cols / RX, h / RY) * 1.6 * ZOOM / Z
    SX = w / 2 - (X - PANX) * s * 2  # quadrant pixels are half a cell wide, so x needs twice the scale
    SY = h / 2 + (Y - PANY) * s

    vis = np.nonzero(NRM @ unrotate(0.0, 0.0, DIST, ca, sa, cp, sp) > PLANE)[0]
    i, j, m = TI[vis, 0], TI[vis, 1], TI[vis, 2]
    depth = Z[i] + Z[j] + Z[m]
    lum = 0.15 + 0.85 * (UNIT[vis] @ unrotate(*LIGHT, ca, sa, cp, sp)).clip(0)
    c = (COL[vis] * lum[:, None]).astype(np.int32)
    color = c[:, 0] << 16 | c[:, 1] << 8 | c[:, 2]

    ax, bx, cx = SX[i], SX[j], SX[m]
    ay, by, cy = SY[i], SY[j], SY[m]
    xi, xj, xm = ax.astype(np.int32), bx.astype(np.int32), cx.astype(np.int32)
    yi, yj, ym = ay.astype(np.int32), by.astype(np.int32), cy.astype(np.int32)
    dot = (xi == xj) & (xi == xm) & (yi == yj) & (yi == ym)
    on = dot & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    off, dep, col = yi[on] * w + xi[on], depth[on], color[on]

    # Triangles wider than a pixel need edge tests. Expanding every bounding box
    # into one flat list of candidate pixels keeps that a numpy job.
    big = np.nonzero(~dot)[0]
    gxi, gxj, gxm = xi[big], xj[big], xm[big]
    gyi, gyj, gym = yi[big], yj[big], ym[big]
    lox = np.minimum(np.minimum(gxi, gxj), gxm).clip(0, w)
    hix = (np.maximum(np.maximum(gxi, gxj), gxm) + 1).clip(0, w)
    loy = np.minimum(np.minimum(gyi, gyj), gym).clip(0, h)
    hiy = (np.maximum(np.maximum(gyi, gyj), gym) + 1).clip(0, h)
    bw, cnt = hix - lox, (hix - lox) * (hiy - loy)
    # Most of them merely straddle one pixel boundary. Painting both ends of such a
    # box skips the expansion entirely and can only spill a sub-pixel sliver.
    p = np.nonzero(cnt == 2)[0]
    tp = big[p]
    off = np.concatenate([off, loy[p] * w + lox[p], (hiy[p] - 1) * w + hix[p] - 1])
    dep = np.concatenate([dep, depth[tp], depth[tp]])
    col = np.concatenate([col, color[tp], color[tp]])
    cnt[p] = 0

    t = np.repeat(np.arange(big.size, dtype=np.int32), cnt)
    loc = np.arange(cnt.sum(), dtype=np.int32) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    px, py = lox[t] + loc % bw[t], loy[t] + loc // bw[t]
    X, Y = px + HALF, py + HALF  # a plain 0.5 would promote the int pixel grid to float64
    t = big[t]  # one fancy index per attribute instead of a select then a gather
    x0, y0, x1, y1, x2, y2 = (v[t] for v in (ax, ay, bx, by, cx, cy))
    e0 = (x1 - x0) * (Y - y0) - (y1 - y0) * (X - x0) < 0
    e1 = (x2 - x1) * (Y - y1) - (y2 - y1) * (X - x1) < 0
    e2 = (x0 - x2) * (Y - y2) - (y0 - y2) * (X - x2) < 0
    ins = np.nonzero((e0 == e1) & (e1 == e2))[0]
    t = t[ins]
    off = np.concatenate([off, py[ins] * w + px[ins]])
    dep = np.concatenate([dep, depth[t]])
    col = np.concatenate([col, color[t]])

    fb = np.zeros(w * h, np.int32)
    order = np.argsort(dep)  # painter's order, so the nearest write lands last
    fb[off[order]] = col[order]

    # Four pixels share a cell but ANSI gives two colours, so split them at the
    # midpoint luminance and average each half.
    quad = fb.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
    chan = np.stack([quad >> 16 & 255, quad >> 8 & 255, quad & 255], -1).astype(float)
    lum = chan @ (0.3, 0.6, 0.1)
    hi = lum > (lum.min(2) + lum.max(2))[:, :, None] / 2
    n = hi.sum(2)[:, :, None]
    pack = (1 << 16, 1 << 8, 1)
    fg = ((chan * hi[:, :, :, None]).sum(2) / np.maximum(n, 1)).astype(np.int64) @ pack
    bg = ((chan * ~hi[:, :, :, None]).sum(2) / np.maximum(4 - n, 1)).astype(np.int64) @ pack
    bits = (hi * (1, 2, 4, 8)).sum(2)
    if TEXT:  # glyph density carries the shading, so the colour can run at full brightness
        mean = chan.mean(2)
        fg = (mean / np.maximum(mean.max(2), 1)[:, :, None] * 255).astype(np.int64) @ pack
        bg = np.zeros_like(fg)
        bits = (lum.mean(2) * (len(RAMP) - 1) / 200).astype(int).clip(0, len(RAMP) - 1)

    out, last = ["\x1b[H"], None
    for frow, brow, grow in zip(fg.tolist(), bg.tolist(), bits.tolist()):
        for t, u, b in zip(frow, brow, grow):
            if (t, u) != last:
                out.append(f"\x1b[38;2;{t >> 16};{t >> 8 & 255};{t & 255};48;2;{u >> 16};{u >> 8 & 255};{u & 255}m")
                last = (t, u)
            out.append(CHARS[b])
        out.append("\x1b[0m\n")
        last = None
    hint = f"\u2190\u2192\u2191\u2193 orbit  +- zoom {ZOOM:.1f}x  wasd pan  space  r reset  q quit"
    out.append(hint[:cols] + "\x1b[K")
    return "".join(out)


detail(160)
sys.stdout.write("\x1b[?25l\x1b[2J")
fd = sys.stdin.fileno()
live = sys.stdin.isatty()
saved = termios.tcgetattr(fd) if live else None
try:
    if live:
        tty.setcbreak(fd)
    t0 = prev = time.monotonic()
    due, ang, spin = 0.0, 0.0, True
    while True:
        while live and select.select([sys.stdin], [], [], 0)[0]:  # arrow keys sit on the Termux extra row
            k = os.read(fd, 64).decode("utf-8", "replace")
            if "q" in k:
                raise KeyboardInterrupt
            if " " in k:
                spin = not spin
            if "r" in k:
                PITCH, YAW, ZOOM, PANX, PANY, spin = 0.4, 0.0, 1.0, 0.0, 0.0, True
            YAW += 0.12 * (k.count("\x1b[D") - k.count("\x1b[C"))
            PITCH = min(1.5, max(-1.5, PITCH + 0.1 * (k.count("\x1b[A") - k.count("\x1b[B"))))
            ZOOM *= 1.25 ** (k.count("+") + k.count("i")) * 0.8 ** (k.count("-") + k.count("o"))
            ZOOM = min(60.0, max(0.25, ZOOM))
            PANX += 0.08 / ZOOM * (k.count("d") - k.count("a"))  # model units, so a step
            PANY += 0.08 / ZOOM * (k.count("w") - k.count("s"))  # always nudges the same distance
            spin = spin and not (k.count("\x1b[C") or k.count("\x1b[D"))
        now = time.monotonic()
        ang, prev = ang + (now - prev) * 0.8 * spin, now
        sys.stdout.write(frame(ang + YAW))
        sys.stdout.flush()
        # Pace at 30 fps, but never build up a backlog when a frame runs long.
        due = max(due + 1 / 30, time.monotonic() - t0)
        time.sleep(max(0.0, t0 + due - time.monotonic()))
except KeyboardInterrupt:
    pass
finally:
    if live:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    sys.stdout.write("\x1b[0m\x1b[?25h\n")
