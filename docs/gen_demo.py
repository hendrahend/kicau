from PIL import Image, ImageDraw, ImageFont
import json, os

W = 480
BG   = (14,21,18);   CARD = (24,36,32);  CARD2 = (31,47,39)
INK  = (238,245,238); DIM = (159,179,164); LINE = (42,59,49)
ACC  = (123,201,111); ACCDK = (12,19,14); AMBER = (232,180,74)

F  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def font(sz, bold=False): return ImageFont.truetype(FB if bold else F, sz)

def rr(d, box, r, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

def wrap(d, text, fnt, maxw):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur+' '+w).strip()
        if d.textlength(t, font=fnt) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

class Page:
    def __init__(self):
        self.img = Image.new('RGB', (W, 4000), BG)
        self.d = ImageDraw.Draw(self.img)
        self.y = 28
        self.pad = 18
    def header(self):
        d, y, p = self.d, self.y, self.pad
        rr(d, [p, y, p+52, y+52], 16, ACC)
        bx, by = p+10, y+12
        d.ellipse([bx, by+8, bx+24, by+26], fill=ACCDK)
        d.ellipse([bx+16, by, bx+32, by+16], fill=ACCDK)
        d.polygon([(bx+32, by+6), (bx+40, by+10), (bx+32, by+13)], fill=ACCDK)
        d.polygon([(bx, by+12), (bx-8, by+4), (bx, by+20)], fill=ACCDK)
        d.ellipse([bx+21, by+5, bx+25, by+9], fill=ACC)
        d.text((p+66, y+2), 'Kicau', font=font(30, True), fill=INK)
        d.text((p+66, y+36), 'Offline bird-call identifier', font=font(14), fill=DIM)
        # language toggle
        for i, (lbl, on) in enumerate([('EN', True), ('ID', False)]):
            bw2 = 44
            x = W - p - (2-i)* (bw2+6) + 6
            rr(d, [x, y+8, x+bw2, y+36], 14, ACC if on else CARD, outline=LINE)
            d.text((x+bw2/2, y+13), lbl, font=font(12, True),
                   fill=ACCDK if on else DIM, anchor='mt')
        self.y = y + 52 + 6
        txt = '100% OFFLINE \u00b7 ON-DEVICE MODEL'
        f = font(12, True); tw = d.textlength(txt, font=f)
        bw = tw + 40
        rr(d, [p, self.y, p+bw, self.y+32], 16, CARD, outline=LINE)
        d.ellipse([p+13, self.y+12, p+21, self.y+20], fill=ACC)
        d.text((p+28, self.y+8), txt, font=f, fill=ACC)
        self.y += 32 + 22
    def tabs(self, active='rec'):
        d, y, p = self.d, self.y, self.pad
        bw = (W - 2*p - 10)//2
        for i, (key, label) in enumerate([('rec','Record'), ('up','Upload')]):
            x = p + i*(bw+10)
            on = key == active
            rr(d, [x, y, x+bw, y+48], 12, ACC if on else CARD, outline=LINE)
            fg = ACCDK if on else INK
            d.text((x+bw//2, y+14), label, font=font(15, True), fill=fg, anchor='mt')
        self.y = y + 48 + 16
    def record_card(self):
        d, y, p = self.d, self.y, self.pad
        rr(d, [p, y, W-p, y+150], 16, CARD, outline=LINE)
        rr(d, [p+20, y+20, W-p-20, y+72], 14, ACC)
        d.text((W//2, y+33), 'Start recording', font=font(17, True), fill=ACCDK, anchor='mt')
        for j, ln in enumerate(wrap(d, 'Record 5\u201330 seconds close to the sound. Early morning, away from roads = best results.', font(13), W-2*p-40)):
            d.text((p+20, y+84+j*20), ln, font=font(13), fill=DIM)
        self.y = y + 150 + 14
    def results(self, data):
        d, y, p = self.d, self.y, self.pad
        d.text((p, y), 'Identification results', font=font(19, True), fill=INK)
        y += 34
        d.text((p, y), 'Recording timeline (per 3 seconds)', font=font(12), fill=DIM)
        y += 24
        n = len(data['chunks']); gap = 4
        segw = (W - 2*p - gap*(n-1)) / n
        for i, c in enumerate(data['chunks']):
            top = c['predictions'][0] if c['predictions'] else None
            x = p + i*(segw+gap)
            if top and top['confidence'] >= 0.05:
                rr(d, [x, y, x+segw, y+38], 7, ACC)
                init = ''.join(w[0] for w in top['common'].replace('-',' ').split()[:2]).upper()
                if segw > 22: d.text((x+segw/2, y+12), init, font=font(11, True), fill=ACCDK, anchor='mt')
            else:
                rr(d, [x, y, x+segw, y+38], 7, CARD2, outline=LINE)
        y += 38 + 16
        for i, s in enumerate(data['summary'][:5]):
            pct = round(s['confidence']*100)
            ch = 118
            rr(d, [p, y, W-p, y+ch], 14, CARD, outline=LINE)
            rr(d, [p+14, y+14, p+48, y+48], 10, CARD2)
            d.text((p+31, y+20), str(i+1), font=font(16, True), fill=AMBER, anchor='mt')
            d.text((p+60, y+12), s['common'], font=font(17, True), fill=INK)
            d.text((p+60, y+36), s['scientific'], font=font(13), fill=DIM)
            rr(d, [p+60, y+62, W-p-16, y+70], 4, ACCDK)
            rr(d, [p+60, y+62, p+60+(W-p-16-p-60)*pct/100, y+70], 4, ACC)
            d.text((p+60, y+78), f'{pct}% confident \u00b7 heard at {s["at"]}s', font=font(12), fill=DIM)
            y += ch + 10
        meta = f'{data["file"]} \u00b7 {data["duration"]} seconds \u00b7 {len(data["chunks"])} segments analyzed'
        d.text((p, y), meta, font=font(12), fill=DIM)
        self.y = y + 30
    def footer(self):
        d, y, p = self.d, self.y, self.pad
        d.line([p, y, W-p, y], fill=LINE, width=1)
        y += 16
        d.text((p, y), 'Field tip:', font=font(13, True), fill=INK)
        y += 24
        lines = wrap(d, 'open this page once while online, keep the tab open \u2014 the model is already on your device, so it keeps working in the forest with zero signal.', font(13), W-2*p)
        for j, ln in enumerate(lines):
            d.text((p, y+j*20), ln, font=font(13), fill=DIM)
        self.y = y + len(lines)*20 + 40
    def save(self, path):
        self.img = self.img.crop((0, 0, W, self.y + 20))
        self.img.save(path)
        print('saved', path, self.img.size)

data = json.load(open('/tmp/tailorbird_en.json'))
here = os.path.dirname(os.path.abspath(__file__))

pg = Page(); pg.header(); pg.tabs(); pg.record_card(); pg.footer()
pg.save(os.path.join(here, 'demo-ui.png'))

pg2 = Page(); pg2.header(); pg2.tabs(); pg2.record_card(); pg2.results(data); pg2.footer()
pg2.save(os.path.join(here, 'demo-result.png'))
