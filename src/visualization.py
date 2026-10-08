from PIL import ImageDraw


def draw_boxes(image, boxes, thickness=4):
    """Draw label-free borders on a copy of the RGB image."""
    if thickness < 1:
        raise ValueError("Border thickness must be at least 1.")
    plotted = image.convert("RGB").copy()
    draw = ImageDraw.Draw(plotted)
    for box in boxes:
        draw.rectangle(tuple(box), outline=(0, 255, 0), width=thickness)
    return plotted
