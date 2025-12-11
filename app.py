import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import io

pd.options.mode.copy_on_write = True

# SITC Classification
SITC_CODES = {
    'TOTAL': 'All Commodities',
    '0': 'Food and Live Animals',
    '2': 'Crude Materials (no fuels)',
    '3': 'Mineral Fuels',
    '5': 'Chemicals & Related Products',
    '6': 'Manufactured Goods',
    '8': 'Miscellaneous Manufactured Articles'
}

# Load trade data
Tfiles = os.listdir('data/Trade/')
Tfiles = sorted(Tfiles, reverse=True)
print(f"Loading: {Tfiles[0]}")

Df = pd.read_csv('data/Trade/' + Tfiles[0],
                  encoding='ISO-8859-1',
                  index_col=False)

# Select and rename columns
features = ['refYear', 'reporterISO', 'reporterDesc',
            'primaryValueX', 'primaryValueM', 'primaryValueBal', 'cmdCode','partnerISO', 'partnerDesc']

Df = Df[features]
Df.rename(columns={'primaryValueX': 'Export',
                   'primaryValueM': 'Import',
                   'primaryValueBal': 'Balance'},
          inplace=True)


def filter_df(df,col,criteria):
    '''filter dataframe according to a column and a criteria
        multiple possible criteria for a column/feature'''
    if not isinstance(criteria,list):
        criteria=[criteria]


    mask=pd.Series(np.full(df.shape[0],False),index=df.index) #add index so  filter can be used on dataframe having specific index (e.g. sub dataframe)
    for i in range(len(criteria)):
        mask+=df[col]==criteria[i]
    out=df[mask]

    return out



def make_network_pct(Df,n_nodes,n_level,start_ISO,commodity,mycol):
#relation 1->2 is not always equal to minus 2->1
# and b) relative importance of relation 1->2 for country 1 (in percentage)     

    Df=filter_df(Df,'partnerISO',iso_ok) #do not consider 'world' as a partner
    queue_ISO=[start_ISO]
    passed_ISO=[]
#    pairs=[]
    Q=pd.DataFrame() # where results are collected

    i=0
    while i<n_level:
        for ISO in list(queue_ISO):
            #print('actual queue',queue_ISO)
            #print('current ISO:', ISO)
            Df_acountry=filter_df(Df,'reporterISO',ISO)
            Df_country_marchandise=filter_df(Df_acountry,'cmdCode',[commodity]) #choos a type of marchandise
            q_sorted=Df_country_marchandise.sort_values(by=mycol,
                                                        key=lambda x: np.abs(x),
                                                        ascending=False)# s
            q_sorted['pct']=q_sorted[mycol].abs()/np.sum(q_sorted[mycol].abs())

            q_sorted_topn=q_sorted.head(n_nodes)
           # print(q_sorted_topn.shape)

            new_ISO=q_sorted_topn['partnerISO'].to_list()
        
            for iso in new_ISO: # add new ISO to the searching list
                queue_ISO.append(iso)

            passed_ISO.append(ISO) #all the country already seen
            for iso in passed_ISO:
                if iso in queue_ISO:
                    queue_ISO.remove(iso)

            Q=pd.concat([Q,q_sorted_topn])
        i=i+1    

    
    G=nx.from_pandas_edgelist(Q[['reporterISO','partnerISO',mycol,'pct']],
                            'reporterISO','partnerISO',
                            edge_attr=[mycol,'pct'],
                            create_using=nx.DiGraph())

    return G


def normalizeNN_pct(G):
    #let suppose a country is only connected those in G.edges(ISO)
    # then relative strenght in percentage is pct/sum(pct)

    for ISO in G.nodes:
        C=dict()
        for edge in G.edges(ISO):
            pct=G.edges[edge]['pct']
            C[edge]=pct

        norm=sum(C.values())

        for key in C.keys():
            G.edges[key]['pct']=C[key]/norm

    return(G)



def compute_rank(G):
    '''
    this is pagerank algorithm but adapted for graph
    and with a wieght on edges
    see https://cs50.harvard.edu/ai/projects/2/pagerank/ for the basic algorithm
    '''
    nodes=G.nodes()
    PR_parent=dict.fromkeys(nodes,1/len(nodes))# initialize
    alpha=.85

    G=normalizeNN_pct(G)
    endnodes=[node for node in G if G.out_degree(node)==0]

    acc=1
    while acc>1e-5:
        PR_child= dict.fromkeys(nodes, 0)# initialize child
        #see https://cs50.harvard.edu/ai/projects/2/pagerank/
        # however this is a bit different, in cs50 ai, rank of a page is
        for iso1 in nodes:
            

            #compute the 2nd right term, this is the 'sum on parent'
            for edge in G.edges(iso1,data='pct'):
                iso2=edge[1]
                wei=edge[2]
                numlink=len(G.edges(iso1))
                #with proba alpha choose one of the edge, which has a weight wei (wei=1 in cs50ai project)
                PR_child[iso2]+=wei*PR_parent[iso1]*alpha/numlink    

            #compute the 1st right term
            PR_child[iso1]+=(1-alpha)*1/len(nodes)# have a change 1-a to choose among all nodes
                    
            ##add additional weight because endnodes exists ; this is not in cs50 ai.. but i think this is correct
            PR_child[iso1]+=alpha/len(nodes)*sum(PR_parent[endnode] for endnode in endnodes)/len(endnodes)

    #
        #normalize child pagerank
        norm=sum(PR_child.values())
        for node in nodes:
            PR_child[node]=PR_child[node]/norm

        err=[]
        for node in nodes:
            dif=abs(PR_parent[node]-PR_child[node])
            err.append(dif)
        acc=sum(err)

        PR_parent=PR_child #update the parent

    return PR_parent





centroids=pd.read_csv('data/country-centroids.csv')#country center location
iso_ok=centroids['alpha3'].to_list()
Df=filter_df(Df,'reporterISO',iso_ok)
Df=filter_df(Df,'partnerISO',iso_ok+['W00']) #W00=the world

# Compute country importance (total trade volume with the world)
Df.loc[:,'Volume']=Df['Export']+Df['Import'] # compute country importance

#remove self interactions on Df
for iso in iso_ok:
    bug=filter_df(filter_df(Df,'reporterISO',iso),'partnerISO',iso)
    if bug.shape[0]>0:
        Df=Df.drop(bug.index,axis=0)




# Initialize Dash app
app = dash.Dash(__name__)


app.layout = html.Div([
    html.Div([
        html.H1("🌍 Global Trade Exploration", style={'textAlign': 'center', 'margin': 0}),
    ], style={'backgroundColor': '#1f77b4', 'color': 'white', 'padding': '20px', 'marginBottom': '20px'}),
    

    # html.Button("Download HTML",
    #                  id="btn_html",
    #                  ),
    # dcc.Download(id="download-html"),

    # Control Panel
    html.Div([
        html.Div([
            html.Label('Commodity:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='commodity-dropdown',
                options=[{'label': v, 'value': k} for k, v in SITC_CODES.items()],
                value='TOTAL',
                clearable=False,
                style={'width': '250px'}
            )
        ], style={'marginBottom': '15px', 'marginRight': '30px', 'display': 'inline-block', 'width': '300px'}),
        
        html.Div([
            html.Label('Metric:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='metric-radio',
                options=[
                    {'label': ' Exports (E)', 'value': 'Export'},
                    {'label': ' Imports (I)', 'value': 'Import'},
                    {'label': ' Volume (I+E)', 'value': 'Volume'},
                    {'label': ' Balance (I-E)', 'value': 'Balance'},
                ],
                value='Balance',
                inline=True,
                style={'display': 'inline-block'}
            )
        ], style={'display': 'inline-block'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px',
              'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Map and Chart container
    html.Div([
        html.Div([
            dcc.Graph(id='world-map')
        ], style={'width': '100%', 'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='top-10-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='top-6-partners-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),

            html.Div([
            #dcc.Graph(id='network-graph')
            html.Img(id='network-graph')
            ], style={'width': '64%', 'display': 'inline-block', 'marginLeft': '-8%'}),

            html.Div([
                dcc.Graph(id='rang-fig')
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '8%'}),

        ])
    ], style={'padding': '20px'})
], style={'padding': '0', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# Callback to update map and chart
@app.callback(
    Output('world-map', 'figure'),
    Output('top-10-chart', 'figure'),
    Output('top-6-partners-chart', 'figure'),
    #Output('network-graph', 'figure'), #not this because I dont know how to create nice image from nx graph in plotly
    Output(component_id='network-graph',component_property= 'src'),
    Output('rang-fig','figure'),
    Input('commodity-dropdown', 'value'),
    Input('metric-radio', 'value'),
    Input('world-map', 'clickData')
)



def update_map(selected_commodity, selected_metric, click_data):
    """Update map based on selected filters and show top n partners when country is clicked"""
    
    year=Df['refYear'].unique()

    # Filter data for selected year and commodity
    #Df_cmd=filter_df(Df,'reporterISO',selected_commodity) #fail! strange
    Df['cmdCode']=Df['cmdCode'].astype(str)
    Df_cmd=Df[Df['cmdCode'].str.startswith(selected_commodity)]
    world_partner=filter_df(Df_cmd,'partnerISO','W00')

    # Determine color scale based on metric
    min=-np.min(world_partner[selected_metric])
    max=np.max(world_partner[selected_metric])
    if selected_metric == 'Balance':
        color_scale = 'RdYlGn'
        #value are can be heavely skewed towards extremes so use symmetric range and min(abs(extremum))
        scope=np.min([min, max])
        rangee=[-scope, scope]

    else:
        color_scale = 'Inferno'
        rangee=[min, max]
    
    # Create choropleth map
    metric_label = {
        'Balance': 'Trade Balance (USD)',
        'Export': 'Exports (USD)',
        'Import': 'Imports (USD)',
        'Volume': 'Trade Volume (USD)'
    }
    
    map_fig = px.choropleth(
        data_frame=world_partner,
        locations='reporterISO',
        color=selected_metric,
        hover_name='reporterDesc',
        range_color=rangee,
        hover_data={
            selected_metric: ':.4g',
            'reporterISO': False
        },
        color_continuous_scale=color_scale,
        title=f"{SITC_CODES[selected_commodity]} - {metric_label[selected_metric]} ({year})"
    )
    
    map_fig.update_layout(
        geo=dict(
            projection_type='natural earth',
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        title_font_size=16,
        font=dict(size=11),
        coloraxis_colorbar=dict(
            title='',#metric_label[selected_metric],
            exponentformat='power',
            ticks='outside',
            orientation='v',
            thickness=15,
            len=0.8
        )
    )
    
    # Create top 10 bar chart
    top_10=world_partner.sort_values(selected_metric,
                    key=lambda x: np.abs(x),
                    ascending=False
                    ).head(10)

    
    bar_fig = px.bar(
        top_10[::-1],
        x=selected_metric,
        y='reporterDesc',
        title=f"Top 10 Countries - {metric_label[selected_metric]}",
        labels={'reporterDesc': 'Country', selected_metric: metric_label[selected_metric]},
        color=selected_metric,
        color_continuous_scale=color_scale,
        range_color=rangee,
        hover_data={selected_metric: ':.4g'},
    )
    
    bar_fig.update_layout(
        height=400,
        margin=dict(l=20, r=50, t=50, b=40),
        showlegend=False,
        font=dict(size=11),
        hovermode='closest',
        coloraxis_colorbar=dict(
            title='',#metric_label[selected_metric],
            exponentformat='power',
            ticks='outside',
            orientation='v',
            thickness=15,
            len=1
        )
    )
    
    bar_fig.update_yaxes(showgrid=False)
    bar_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Create top 3 trading partners chart
    selected_country = None
    if click_data and 'points' in click_data and len(click_data['points']) > 0:
        selected_country = click_data['points'][0].get('hovertext')
    
    if selected_country:        
        # Get partners for selected country
        country_iso = Df[Df['reporterDesc'] == selected_country]['reporterISO'].values
        if len(country_iso) > 0:
            country_iso = country_iso[0]
            partners_data = Df[Df['reporterISO'] == country_iso].copy()
            partners_data = partners_data[partners_data['partnerISO'] != 'W00']  # Exclude world total
            
            # Aggregate by partner
            partners_agg = partners_data.groupby(['partnerISO', 'partnerDesc']).agg({
                'Export': 'sum',
                'Import': 'sum',
                'Volume': 'sum',
                'Balance': 'sum'
            }).reset_index()
            
            # Get top n
            n=6
            top_n = partners_agg.sort_values(selected_metric,
                                            key=lambda x: np.abs(x), 
                                            ascending=False
                                            ).head(n)

            partners_fig = px.bar(
                top_n[::-1],
                x=selected_metric,
                y='partnerDesc',
                title=f"Top {n} Trading Partners of {selected_country} - {metric_label[selected_metric]}",
                labels={'partnerDesc': 'Partner Country', selected_metric: metric_label[selected_metric]},
                color=selected_metric,
                color_continuous_scale=color_scale,
                range_color=rangee,
                hover_data={selected_metric: ':.4g'},
            )
        else:
            partners_fig = go.Figure().add_annotation(text="Select a country from the map")
    else:
        partners_fig = go.Figure().add_annotation(text="Click on a country to see top trading partners")
    
    partners_fig.update_layout(
        height=400,
        margin=dict(l=20, r=50, t=50, b=40),
        showlegend=False,
        font=dict(size=11),
        hovermode='closest',
        coloraxis_colorbar=dict(
            title='',#metric_label[selected_metric],
            exponentformat='power',
            ticks='outside',
            orientation='v',
            thickness=15,
            len=1
        )
    )
    
    partners_fig.update_yaxes(showgrid=False)
    partners_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    
    
    #network graph figure

    selected_country_iso =Df[Df['reporterDesc'] == selected_country]['reporterISO'].values
    if len(selected_country_iso) > 0:
        selected_country_iso = selected_country_iso[0]
    else:
        selected_country_iso = None

    G=make_network_pct(Df=Df,
                       n_nodes=4,
                       n_level=2,
                       start_ISO=selected_country_iso,
                       commodity=selected_commodity,
                       mycol=selected_metric)
    
    # Example graph
    #G = nx.gnp_random_graph(10, 0.3, seed=42) 
    #G_pos = nx.spring_layout(G, k=0.25, iterations=10)
    #for node in G.nodes():
    #    G.nodes[node]['pos'] = G_pos[node]
    

    #if len(G.nodes())>0:
    if selected_country:

        start_ISO=list(G.nodes)[0] 
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot',root=start_ISO)

        #if cmap is None:
        cmap=plt.cm.coolwarm

        Size=[G.edges[pair][selected_metric] for pair in list(G.edges())]
        min_val=np.min(np.abs(Size))
        max_val=np.max(np.abs(Size))
        Size=np.abs(Size)/max_val
        Size=np.log10(Size)

        fig,ax=plt.subplots(figsize=(10,6))
        nx.draw_networkx(G,
                            edge_cmap=cmap,
                            edge_color=Size,
                            node_color='k',
                            node_size=10,
                            width=2,
                            pos=pos,
                            connectionstyle='arc3,rad=.3'# connection style in matplotlib
                            )

        ### add attribute 'pct' in edges
        attr='pct'
        labels={}
        for edge in list(G.edges()):
            labels[edge]='{:2.0f}'.format(G.edges[edge][attr]*100)+'%'

        #color can NOT be a list so one must run a loop for every edge
        for edge in list(G.edges()):
            color='k'
            if G.edges[edge][selected_metric]<0:
                color='r'
            nx.draw_networkx_edge_labels(G,pos=pos,
                                    edge_labels={edge:labels[edge]},
                                    connectionstyle='arc3,rad=0.3',
                                    font_color=color
                                    )


        color_norm = mpl.colors.LogNorm(vmin=min_val, vmax=max_val)
        fig.colorbar(mpl.cm.ScalarMappable(norm=color_norm,cmap=cmap),
                    ax=ax,orientation='horizontal',
                    label=f'Edge color scale for {selected_metric} (USD)',
                    fraction=0.046, pad=0.04,#ticks=[min_val, max_val],
                    #ticklabel=['{:.0f}'.format(min_val),'{:.0f}'.format(max_val)]  
                    )
        
        #cbar.ax.locator_params(nbins=5)
        #mpl.ticker.LogLocator(base=10.0, numticks=4)
#        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))
        #ax.text(0.0, 0.1, "LogLocator(base=10, numticks=15)",
        #fontsize=15, transform=ax.transAxes)

        ax.set_title(f"Trade Network for {selected_country} - {SITC_CODES[selected_commodity]}")
        plt.axis('off') 

    else:
        fig,ax=plt.subplots(figsize=(12,6))
        ax.set_title(f"Click on a country to see network graph")
        plt.axis('off')    

    # Convert Matplotlib figure to PNG image and encode in base64
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")#.replace("\n", "")
    netw_fig = f'data:image/png;base64,{fig_data}'


    if selected_country_iso:
        ISOrank=compute_rank(G)
    #    L=sorted(ISOrank,key=ISOrank.get,reverse=True)       
    #    for l in L:
    #        print(l,'{:.2f}'.format(ISOrank[l]*100)+'%',Df_cmd[])

        pie_df=pd.DataFrame(ISOrank.items(),columns=['ISO','Rank'])

        rank_fig=px.pie(data_frame=pie_df,
                names='ISO',
                values='Rank',
                title=f'Country Rank in Trade Network for {selected_country}',# \\ \n {SITC_CODES.get(selected_commodity, selected_commodity)})",
                labels='ISO',
                hole=.3)
        
        rank_fig.update_traces(textposition='inside', textinfo='percent+label')

    else:
        rank_fig = go.Figure().add_annotation(text="Select a country from the map")


    rank_fig.update_layout(
        height=500,
        margin=dict(l=20, r=50, t=40, b=80),
        showlegend=False,
        font=dict(size=11),
        hovermode='closest',
        coloraxis_colorbar=dict(
            title='',#metric_label[selected_metric],
            ticks='outside',
            orientation='v',
            thickness=15,
            len=1
        )
    )

    return map_fig, bar_fig, partners_fig, netw_fig,rank_fig

# buffer = io.StringIO()
# html_bytes = buffer.getvalue().encode()
# encoded = base64.b64encode(html_bytes).decode()

# app.layout = html.Div([
#     html.H4('Simple plot export options'),
#     html.P("↓↓↓ try downloading the plot as PNG ↓↓↓", style={"text-align": "right", "font-weight": "bold"}),
#     dcc.Graph(id="graph", figure=fig),
#     html.A(
#         html.Button("Download as HTML"),
#         id="download",
#         href="data:text/html;base64," + encoded,
#         download="plotly_graph.html"
#     )
# ])



if __name__ == '__main__':
    # Add this after the if __name__ == '__main__': block

    @app.callback(
        Output('download-html', 'data'),
        Input('download-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_html(n_clicks):
        return dcc.send_file('app.html')
    
    app.run(debug=True)
