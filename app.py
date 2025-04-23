
from flask import Flask, render_template
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import io, base64

app = Flask(__name__)

# --- Load & preprocess data ---
df = pd.read_csv('static/Results_21Mar2022.csv')

def categorize_diet(diet):
    if 'vegan' in diet:
        return 'Plant-based'
    elif 'veggie' in diet:
        return 'Vegetarian'
    elif 'meat100' in diet:
        return 'High meat consumption'
    elif 'meat50-99' in diet:
        return 'Medium meat consumption'
    elif 'fish' in diet:
        return 'Pescatarian'
    else:
        return 'Other'

def categorize_age(age_str):
    age = int(age_str.split('-')[0])
    if age <= 29: return 'Young adult'
    if age <= 39: return 'Adult'
    if age <= 49: return 'Mature adult'
    if age <= 59: return 'Senior adult'
    return 'Elderly'

def to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def identify_highly_correlated(df, threshold=0.9, exclude_outliers=[]):
    corr_matrix = df.corr().abs()
    high_corr_pairs = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold and corr_matrix.columns[j] not in exclude_outliers:
                high_corr_pairs.add((corr_matrix.columns[i], corr_matrix.columns[j]))

    to_remove = set()
    for col1, col2 in high_corr_pairs:
        if col1 not in exclude_outliers:
            to_remove.add(col1)
        if col2 not in exclude_outliers:
            to_remove.add(col2)

    return high_corr_pairs, to_remove

df['Diet Category'] = df['diet_group'].apply(categorize_diet)
df['Age Category']  = df['age_group'].apply(categorize_age)

mean_metrics = [
    'mean_ghgs','mean_land','mean_watscar','mean_eut',
    'mean_ghgs_ch4','mean_ghgs_n2o','mean_bio','mean_watuse','mean_acid'
]

# Correlation-based reduction
numeric_df = df[mean_metrics]
high_corr_pairs, suggested_removals = identify_highly_correlated(numeric_df, threshold=0.9, exclude_outliers=['mean_land'])
df_reduced = df.drop(columns=list(suggested_removals))

# --- Treemap ---
fig_tm = px.treemap(
    df, path=['Diet Category','Age Category','sex'],
    values='mean_land', color='mean_land',
    color_continuous_scale='RdBu',
    color_continuous_midpoint=np.average(df['mean_land'], weights=df['n_participants']),
    title='Mean Land Use by Diet → Age → Sex'
)
fig_tm.data[0].textinfo = 'label+value'

# --- Parallel Coordinates ---
df_pc = df.groupby('Diet Category', observed=True)[mean_metrics].mean().reset_index()
df_pc[mean_metrics] = MinMaxScaler().fit_transform(df_pc[mean_metrics])
codes = {c:i for i,c in enumerate(df_pc['Diet Category'])}
df_pc['code'] = df_pc['Diet Category'].map(codes)
fig_pc = go.Figure(go.Parcoords(
    line=dict(color=df_pc['code'], colorscale='Plotly3', showscale=True,
              colorbar=dict(tickmode='array', tickvals=list(codes.values()), ticktext=list(codes.keys()))),
    dimensions=[dict(label=m.replace('mean_','').upper(), values=df_pc[m]) for m in mean_metrics]
))
fig_pc.update_layout(title='Normalized Environmental Profiles')

# --- Radar Chart ---
df_rb = df.groupby('Diet Category', observed=True)[mean_metrics].mean().reset_index()
df_melt = df_rb.melt(id_vars='Diet Category', var_name='Metric', value_name='Value')
fig_radar = px.line_polar(df_melt, r='Value', theta='Metric', color='Diet Category',
                          line_close=True, color_discrete_sequence=px.colors.qualitative.Plotly)
fig_radar.update_traces(fill='toself').update_layout(title='Radar Chart of Diet Impacts')

# --- Correlation Network ---
corr = df[mean_metrics].corr().abs()
G = nx.Graph()
for i in mean_metrics:
    for j in mean_metrics:
        if i<j and corr.loc[i,j]>0.6:
            G.add_edge(i.replace('mean_',''), j.replace('mean_',''), weight=corr.loc[i,j])
pos = nx.spring_layout(G, seed=42)
edge_x, edge_y = [], []
for u,v in G.edges():
    x0,y0=pos[u]; x1,y1=pos[v]
    edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
edge = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='rgba(100,100,100,0.5)'))
node = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                  mode='markers+text', text=list(G.nodes()), textposition='bottom center',
                  marker=dict(size=[8+4*G.degree(n) for n in G.nodes()],
                              color=px.colors.qualitative.Vivid[:len(G.nodes())]))
fig_net = go.Figure([edge,node]).update_layout(title='Correlation Network')

# --- Heatmap PNGs ---
buf1 = io.BytesIO()
plt.figure(figsize=(6,5))
sns.heatmap(corr, cmap='viridis', annot=True, fmt='.2f')
plt.tight_layout(); plt.savefig(buf1, format='png'); plt.close()
heatmap_b64 = base64.b64encode(buf1.getvalue()).decode()

link = sch.linkage(squareform(1-corr.values), method='average')
order = sch.dendrogram(link, labels=[m.replace('mean_','') for m in mean_metrics], no_plot=True)['ivl']
mat = corr.copy()
mat.index = [m.replace('mean_','') for m in mean_metrics]; mat.columns = mat.index
buf2 = io.BytesIO()
plt.figure(figsize=(6,6))
sns.heatmap(mat.loc[order,order], cmap='mako', annot=True, fmt='.2f')
plt.tight_layout(); plt.savefig(buf2, format='png'); plt.close()
clustermap_b64 = base64.b64encode(buf2.getvalue()).decode()

# --- Scatter Matrix & PCA ---
fig_sp = px.scatter_matrix(df, dimensions=mean_metrics[:5], color='Diet Category',
                           color_discrete_sequence=px.colors.qualitative.T10)
fig_sp.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.7)).update_layout(title='Scatterplot Matrix')

coords = PCA(n_components=2).fit_transform(df[mean_metrics])
df_pca = pd.DataFrame(coords, columns=['PC1','PC2']); df_pca['Diet Category'] = df['Diet Category']
fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Diet Category',
                     color_discrete_sequence=px.colors.qualitative.Safe).update_layout(title='PCA Biplot')

# --- Sankey Diagram ---
df['GHG Level'] = pd.qcut(df['mean_ghgs'],3,labels=['Low','Medium','High'])
df_s = df.groupby(['Diet Category','GHG Level'], observed=True).size().reset_index(name='count')
diet_cols = {'Plant-based':'rgb(102,194,165)','Vegetarian':'rgb(252,141,98)',
             'Pescatarian':'rgb(141,160,203)','Medium meat consumption':'rgb(231,138,195)',
             'High meat consumption':'rgb(166,216,84)','Other':'rgb(255,217,166)'}
ghg_cols = {'Low':'rgb(255,255,204)','Medium':'rgb(161,218,180)','High':'rgb(65,182,196)'}
nodes = list(diet_cols)+list(ghg_cols)
src = df_s['Diet Category'].map({n:i for i,n in enumerate(nodes)})
tgt = df_s['GHG Level'].map({n:i for i,n in enumerate(nodes)})
fig_sankey = go.Figure(go.Sankey(
    node=dict(label=nodes, color=[diet_cols.get(n,ghg_cols.get(n)) for n in nodes]),
    link=dict(source=src, target=tgt, value=df_s['count'],
              color=df_s['Diet Category'].map(diet_cols))
)).update_layout(title='Sankey Diagram')

@app.route('/')
def index():
    return render_template('index.html',
        tm_div=to_div(fig_tm),
        pc_div=to_div(fig_pc),
        radar_div=to_div(fig_radar),
        net_div=to_div(fig_net),
        sp_div=to_div(fig_sp),
        pca_div=to_div(fig_pca),
        sankey_div=to_div(fig_sankey),
        heatmap_b64=heatmap_b64,
        clustermap_b64=clustermap_b64
    )

if __name__=='__main__':
    app.run()