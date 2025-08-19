import shap
import matplotlib.pyplot as plt
import os

def explain_model(model, X, figures_dir="reports/figures"):
    """
    Calcula SHAP values y genera gráfico summary plot
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    fig = shap.summary_plot(shap_values[1], X, show=False)
    
    # Guardar figura
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_path = os.path.join(figures_dir, "shap_summary.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    print(f"SHAP figure guardada en: {fig_path}")
