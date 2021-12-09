from . import polygon_geo_cpu


def polygon_iou(poly1, poly2):
    """Compute the IoU of polygons."""
    return polygon_geo_cpu.polygon_iou(poly1, poly2)
