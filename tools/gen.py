import json, sys, os
from xml.sax.saxutils import escape

INST = json.load(open('inst.json')) if os.path.exists('inst.json') else {}

class Circ:
    def __init__(s, name):
        s.name = name; s.comps = []; s.wires = []; s.ins = []; s.outs = []
    def comp(s, lib, name, loc, **attrs):
        s.comps.append((lib, name, loc, attrs))
    def wire(s, p, q):
        if p != q: s.wires.append((p, q))
    def tun(s, pt, label, width, side, d=20):
        x, y = pt
        if side == 'L':   q, f = (x - d, y), 'east'
        elif side == 'R': q, f = (x + d, y), 'west'
        elif side == 'D': q, f = (x, y + d), 'north'
        elif side == 'U': q, f = (x, y - d), 'south'
        s.wire(pt, q)
        s.comp(0, 'Tunnel', q, label=label, width=width, facing=f)
    def const(s, pt, val, width, side='L', d=20):
        x, y = pt
        if side == 'L':
            q = (x - d, y); f = 'east'
        else:
            q = (x, y + d); f = 'north'
        s.wire(pt, q)
        s.comp(0, 'Constant', q, value=hex(val), width=width, facing=f)
    def pin_in(s, loc, label, width, tunlabel=None):
        s.ins.append(label)
        s.comp(0, 'Pin', loc, label=label, width=width, appearance='NewPins')
        s.tun(loc, tunlabel or label, width, 'R')
    def pin_out(s, loc, label, width, tunlabel=None):
        s.outs.append(label)
        s.comp(0, 'Pin', loc, label=label, width=width, output='true', facing='west', appearance='NewPins')
        s.tun(loc, tunlabel or label, width, 'L')
    # ---- primitives (all facing east) ----
    def gate(s, kind, loc, ins, out, width=1):
        x, y = loc
        n = len(ins)
        if n == 2:
            dx = 40 if kind == 'XOR Gate' else 30
            offs = [(-dx, -10), (-dx, 10)]
            attrs = dict(size='30')
        else:
            assert n == 8
            offs = [(-50, d) for d in (-40, -30, -20, -10, 10, 20, 30, 40)]
            attrs = dict(inputs='8')
        if width > 1: attrs['width'] = width
        s.comp(1, kind, loc, **attrs)
        for (dx, dy), lab in zip(offs, ins):
            p = (x + dx, y + dy)
            if isinstance(lab, int): s.const(p, lab, width)
            else: s.tun(p, lab, width, 'L')
        s.tun(loc, out, width, 'R')
    def notg(s, loc, i, o, width):
        x, y = loc
        s.comp(1, 'NOT Gate', loc, width=width)
        s.tun((x - 30, y), i, width, 'L'); s.tun(loc, o, width, 'R')
    def split(s, loc, whole, bits, side):
        """8-bit splitter; bits[k] = tunnel label, int constant, or None."""
        x, y = loc
        s.comp(0, 'Splitter', loc, fanout=8, incoming=8)
        s.tun(loc, whole, 8, side)
        for k, lab in enumerate(bits):
            p = (x + 20, y - 80 + 10 * k)
            if lab is None: continue
            if isinstance(lab, int):
                s.wire(p, (p[0] + 20, p[1]))
                s.comp(0, 'Constant', (p[0] + 20, p[1]), value=hex(lab), facing='west')
            else:
                s.tun(p, lab, 1, 'R')
    def sub(s, cname, loc, ins, outs):
        off = INST[cname]
        x, y = loc
        s.comp(None, cname, loc)
        for (dx, dy, w), lab in zip(off['in'], ins):
            p = (x + dx, y + dy)
            if isinstance(lab, int): s.const(p, lab, w)
            else: s.tun(p, lab, w, 'L')
        for (dx, dy, w), lab in zip(off['out'], outs):
            if lab is None: continue
            s.tun((x + dx, y + dy), lab, w, 'R')

    def xml(s):
        o = [f'<circuit name="{s.name}">', f'<a name="circuit" val="{s.name}"/>']
        for p, q in s.wires:
            o.append(f'<wire from="({p[0]},{p[1]})" to="({q[0]},{q[1]})"/>')
        for lib, name, loc, attrs in s.comps:
            l = '' if lib is None else f' lib="{lib}"'
            o.append(f'<comp{l} loc="({loc[0]},{loc[1]})" name="{name}">')
            for k, v in attrs.items():
                o.append(f'  <a name="{k}" val="{escape(str(v))}"/>')
            o.append('</comp>')
        o.append('</circuit>')
        return '\n'.join(o)

def full_adder():
    c = Circ('FullAdder')
    c.pin_in((40, 60), 'a', 1); c.pin_in((40, 100), 'b', 1); c.pin_in((40, 140), 'cin', 1)
    c.pin_out((460, 80), 's', 1); c.pin_out((460, 160), 'cout', 1)
    c.gate('XOR Gate', (200, 60), ['a', 'b'], 'x')
    c.gate('XOR Gate', (360, 80), ['x', 'cin'], 's')
    c.gate('AND Gate', (200, 160), ['a', 'b'], 'g')
    c.gate('AND Gate', (200, 240), ['x', 'cin'], 'p')
    c.gate('OR Gate', (360, 180), ['g', 'p'], 'cout')
    return c

def adder8():
    c = Circ('Adder8')
    c.pin_in((40, 100), 'A', 8); c.pin_in((40, 200), 'B', 8); c.pin_in((40, 300), 'Cin', 1, 'c0')
    c.pin_out((760, 100), 'S', 8); c.pin_out((760, 200), 'Cout', 1, 'c8')
    c.split((120, 520), 'A', [f'a{k}' for k in range(8)], 'L')
    c.split((120, 720), 'B', [f'b{k}' for k in range(8)], 'L')
    c.split((120, 920), 'S', [f's{k}' for k in range(8)], 'L')
    for k in range(8):
        c.sub('FullAdder', (360 + 200 * (k % 4), 420 + 160 * (k // 4) + 400 * 0), [f'a{k}', f'b{k}', f'c{k}'], [f's{k}', f'c{k+1}'])
    return c

def subtractor8():
    c = Circ('Subtractor8')
    c.pin_in((40, 100), 'A', 8); c.pin_in((40, 200), 'B', 8)
    c.pin_out((560, 100), 'D', 8); c.pin_out((560, 200), 'Cout', 1)
    c.notg((200, 320), 'B', 'nB', 8)   # A - B = A + ~B + 1
    c.sub('Adder8', (380, 420), ['A', 'nB', 1], ['D', 'Cout'])
    return c

def shift_right():
    c = Circ('ShiftRight')
    c.pin_in((40, 100), 'B', 8)
    c.pin_out((460, 100), 'O', 8); c.pin_out((460, 160), 'Cout', 1, 'b0')
    c.split((120, 340), 'B', [f'b{k}' for k in range(8)], 'L')
    c.split((320, 340), 'O', [f'b{k+1}' for k in range(7)] + [0], 'L')
    return c

def multiplier():
    c = Circ('Multiplier4x4')
    c.pin_in((40, 100), 'A', 8); c.pin_in((40, 200), 'B', 8)
    c.pin_out((1260, 100), 'P', 8)
    c.split((120, 420), 'A', [f'a{k}' for k in range(4)] + [None] * 4, 'L')
    c.split((120, 620), 'B', [f'b{k}' for k in range(4)] + [None] * 4, 'L')
    # partial products pp_i_j = a_i AND b_j
    for j in range(4):
        for i in range(4):
            c.gate('AND Gate', (340 + 140 * i, 360 + 80 * j), [f'a{i}', f'b{j}'], f'pp{i}_{j}')
    # row words W_j = (A[3:0] & b_j) << j
    for j in range(4):
        bits = [f'pp{k-j}_{j}' if 0 <= k - j < 4 else 0 for k in range(8)]
        c.split((340 + 220 * j, 800), f'W{j}', bits, 'L')
    c.sub('Adder8', (500, 960), ['W0', 'W1', 0], ['S1', None])
    c.sub('Adder8', (800, 960), ['S1', 'W2', 0], ['S2', None])
    c.sub('Adder8', (1100, 960), ['S2', 'W3', 0], ['P', None])
    return c

def mux8(width):
    c = Circ(f'Mux8to1_{width}bit')
    for k in range(8):
        c.pin_in((40, 60 + 60 * k), f'I{k}', width)
    c.pin_in((40, 560), 'Sel', 3)
    c.pin_out((760, 60), 'Out', width)
    x, y = 200, 760
    c.comp(2, 'Decoder', (x, y), select=3, enable='false')
    c.tun((x, y), 'Sel', 3, 'D')
    for k in range(8):
        c.tun((x + 20, y - 80 + 10 * k), f'd{k}', 1, 'R')
    for k in range(8):
        en = f'd{k}'
        if width > 1:
            c.comp(0, 'Bit Extender', (400, 620 + 50 * k), in_width=1, out_width=width, type='sign')
            c.tun((360, 620 + 50 * k), f'd{k}', 1, 'L')
            c.tun((400, 620 + 50 * k), f'e{k}', width, 'R')
            en = f'e{k}'
        c.gate('AND Gate', (400, 80 + 60 * k), [f'I{k}', en], f'm{k}', width)
        # (gate inputs are tunnels, so I/en labels connect)
    c.gate('OR Gate', (640, 280), [f'm{k}' for k in range(8)], 'Out', width)
    return c

def alu():
    c = Circ('main')
    c.pin_in((40, 100), 'A', 8); c.pin_in((40, 180), 'B', 8); c.pin_in((40, 260), 'Opcode', 3)
    c.pin_out((1100, 100), 'O', 8); c.pin_out((1100, 180), 'Cout', 1)
    # 000: O = B  -> r0 is just B
    c.gate('XOR Gate', (280, 120), ['A', 'B'], 'r1', 8)   # 001
    c.gate('AND Gate', (280, 200), ['A', 'B'], 'r2', 8)   # 010
    c.gate('OR Gate',  (280, 280), ['A', 'B'], 'r3', 8)   # 011
    c.sub('Adder8', (280, 400), ['A', 'B', 0], ['r4', 'c4'])          # 100
    c.sub('Subtractor8', (280, 520), ['A', 'B'], ['r5', 'c5'])        # 101
    c.sub('ShiftRight', (280, 620), ['B'], ['r6', 'c6'])              # 110
    c.sub('Multiplier4x4', (280, 720), ['A', 'B'], ['r7'])            # 111
    c.sub('Mux8to1_8bit', (760, 300), ['B', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'Opcode'], ['O'])
    c.sub('Mux8to1_1bit', (760, 700), [0, 0, 0, 0, 'c4', 'c5', 'c6', 0, 'Opcode'], ['Cout'])
    return c

LIBS = '''<lib desc="#Wiring" name="0"><tool name="Pin"><a name="appearance" val="NewPins"/></tool></lib>
<lib desc="#Gates" name="1"/>
<lib desc="#Plexers" name="2"/>
<lib desc="#Arithmetic" name="3"/>
<lib desc="#Memory" name="4"/>
<lib desc="#I/O" name="5"/>
<lib desc="#Base" name="6"/>'''

def write(path, circs, main):
    with open(path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<project source="3.9.0" version="1.0">\n')
        f.write(LIBS + f'\n<main name="{main}"/>\n<options><a name="simlimit" val="10000"/></options>\n')
        for c in circs: f.write(c.xml() + '\n')
        f.write('</project>\n')

basics = [full_adder, adder8, subtractor8, shift_right, multiplier, lambda: mux8(8), lambda: mux8(1)]
if sys.argv[1] == 'probe':
    built = []
    probe = Circ('probe')
    y = 100
    for fn in basics:
        try: c = fn()
        except (KeyError, TypeError): continue
        built.append(c)
        probe.comp(None, c.name, (500, y)); y += 400
    pins = {c.name: (c.ins, c.outs) for c in built}
    json.dump(pins, open('pins.json', 'w'))
    write('probe.circ', built + [probe], 'probe')
else:
    circs = [alu()] + [fn() for fn in basics]
    write(sys.argv[2], circs, 'main')
