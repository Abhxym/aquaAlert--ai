import os
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

@router.get("/plots")
def list_plots():
    result = []
    groups = {
        "EDA": [
            ("Correlation Heatmap",      "plot1_heatmap.png"),
            ("Feature Distribution",     "plot2_distribution.png"),
            ("Scatter Analysis",         "plot3_scatter.png"),
            ("Class Balance",            "plot4_class_balance.png"),
            ("Outlier Detection",        "plot5_outliers.png"),
        ],
        "Clustering": [
            ("Elbow Curve",              "clustering/plot1_elbow.png"),
            ("Silhouette Score",         "clustering/plot2_silhouette.png"),
            ("Spatial Clusters",         "clustering/plot3_spatial.png"),
            ("Meteorological Clusters",  "clustering/plot4_meteo.png"),
            ("Cluster Validation",       "clustering/plot5_validation.png"),
            ("Violin Distribution",      "clustering/plot6_violin.png"),
        ],
    }
    for group, items in groups.items():
        for title, filename in items:
            full = os.path.join(PLOTS_DIR, filename)
            if os.path.exists(full):
                result.append({"group": group, "title": title, "file": filename})
    return JSONResponse(result)

@router.get("/plots/image")
def get_plot(file: str):
    safe = os.path.normpath(file).lstrip("/\\")
    full = os.path.join(PLOTS_DIR, safe)
    if not full.startswith(PLOTS_DIR) or not os.path.exists(full):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(full, media_type="image/png")
