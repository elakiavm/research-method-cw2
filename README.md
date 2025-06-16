
# Diet & Environmental Impact Dashboard ( MAIN FILE NAME : main.ipynb)

This repository provides an interactive data visualization dashboard that explores the environmental impacts of various dietary patterns in the UK. The project is built using Python, and all analysis and plots are implemented in a Jupyter Notebook, enabling transparent, reproducible exploration.

---

## Repository Structure

```
Research_Methods_CW2/
files for website
├── static/
│   └── Results_21Mar2022.csv
├── templates/
│   └── index.html
├── README.md
└── app.py
main file 
├── main.ipynb 
```

---

## About the Dataset

The dataset originates from the [EPIC-Oxford study](https://ora.ox.ac.uk/objects/uuid:ca441840-db5a-48c8-9b82-1ec1d77c2e9c), which investigates how different diets (vegan, vegetarian, fish-eater, and meat-eater) impact the environment. 

It includes:
- Environmental impact metrics: **GHG emissions, land use, water scarcity, eutrophication, biodiversity loss**, etc.
- Demographics: **sex, age group, diet group**
- Data standardized to **2,000 kcal/day** with **Monte Carlo simulations (n=1000)** to ensure robustness

---

## 📘 How to Use This Notebook(Diet_Impact_Dashboard.ipynb)

### Step-by-Step Workflow

1. **Import Libraries**  
   Load all required packages including Plotly, Seaborn, NetworkX, and Scikit-learn.

2. **Load Dataset**  
   Import the CSV file `Results_21Mar2022.csv` from the `Dataset/` folder.

3. **Preprocessing**
   - Categorize diet and age groups.
   - Apply correlation analysis to reduce redundant features.

4. **Visualizations**
   - 🌍 Treemap (Land Use by Diet, Age & Sex)  
   - 📈 Parallel Coordinates (Multi-metric comparison)  
   - 🔥 Heatmaps (Raw + Clustered correlations)  
   - 🧭 Radar Chart (Impact profiles by diet)  
   - 🔗 Correlation Network (r > 0.6 links)  
   - 🟦 Scatterplot Matrix (Pairwise metric analysis)  
   - 🎯 PCA Biplot (Dimensionality reduction)  
   - 🔄 Sankey Diagram (Diet → GHG level flow)

5. **Export & Save**  
   Heatmaps and HTML-based visualizations can be saved directly from the notebook.

---

## Recommended Usage

For the best experience, open the notebook using **Google Colab** or **JupyterLab**:

- Colab provides full interactivity for Plotly graphs.
- Local Jupyter Notebooks will render static plots unless running on supported environments.

🔗 [Open in Google Colab (Recommended)](https://github.com/elakiavm/research-method-cw2/blob/main/Diet_Impact_Dashboard.ipynb)

---

## Insights Highlighted

- Plant-based diets consistently show the **lowest environmental footprint** across all metrics.
- A unique treemap pattern reveals that **land use in high meat consumers doesn't scale linearly with age**, suggesting complex lifestyle-diet interactions.
- Feature reduction via correlation analysis improves visualization clarity and supports dimensionality reduction techniques like PCA.

---

## Links

- [Dataset Source (Oxford ORA)](https://ora.ox.ac.uk/objects/uuid:ca441840-db5a-48c8-9b82-1ec1d77c2e9c)
- [Interactive Dashboard ](https://research-method-cw2.onrender.com/)
- [Project Code Repository](https://github.com/elakiavm/-Research_Methods_CW2.git)

