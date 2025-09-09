
from PIL import Image, ImageDraw, ImageFont
import os, datetime

def make_card(result: dict, out_dir="static/results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    W, H = 600, 400
    img = Image.new("RGB", (W, H), (245, 245, 250))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_body = ImageFont.truetype("arial.ttf", 18)
    except:
        font_title = font_body = None

    # 제목
    title = "AI 음성 성격 프로파일"
    draw.text((20, 20), title, fill="black", font=font_title)

    # 설명
    y = 80
    for line in result["description"].split("\n"):
        draw.text((20, y), line, fill="black", font=font_body)
        y += 30

    # 태그
    draw.text((20, y+20), "Tags: " + ", ".join(result.get("tags", [])), fill="blue", font=font_body)

    filename = f"card_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(out_dir, filename)
    img.save(path)
    return path
