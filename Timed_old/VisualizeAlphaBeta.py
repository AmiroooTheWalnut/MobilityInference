
import pandas as pd
import numpy as np
import io
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import torch
import torch.distributions.constraints as constraints
import pyro

def showAlphaBetaRange(type,data,needsVerbose):
    fig = make_subplots(rows=3, cols=1)
    # fig = go.Figure()

    ageOccupation=[]
    for i in range(data.occupationProb.shape[1]):
        for j in range(data.occupationProb.shape[0]):
            ageOccupation.append(needsVerbose["age range"][i+j*data.occupationProb.shape[1]]+needsVerbose["Occupation"][i])
    # ageOccupation=[''.join(i) for i in zip(needsVerbose["age range"].map(str),needsVerbose["Occupation"])]
    thresh = torch.div(data.alpha_paramShop, data.alpha_paramShop + data.beta_paramShop)
    if type == 'CBG based simulation':
        avgCPU=(data.BBNSh.sum(-1) / data.BBNSh.shape[1]).cpu()
        thresh=torch.mul(thresh, avgCPU)
        xLim=data.BBNSh.sum(-1)/data.BBNSh.shape[1]
        xLim=xLim.tolist()
    else:
        xLim=needsVerbose['Retailing services']
        
    thresh=thresh.tolist()

    threshStr = [str(s) for s in thresh]

    for i in range(len(needsVerbose)):
        fig.add_trace(go.Scatter(
            x=[0, xLim[i]],
            y=[ageOccupation[i], ageOccupation[i]],
            orientation='h',
            line=dict(color='rgb(244,165,130)', width=8),
        ),row=1, col=1)

    fig.add_trace(go.Scatter(
        x=thresh,
        y=ageOccupation,
        marker=dict(color='#CC5700', size=14),
        mode='markers+text',
        text=threshStr,
        textposition='middle left',
        name='alpha/(alpha+beta)'),row=1, col=1)

    fig.update_layout(title=type, showlegend=False)

    # fig.show()

    # fig = go.Figure()

    ageOccupation = []
    for i in range(data.occupationProb.shape[1]):
        for j in range(data.occupationProb.shape[0]):
            ageOccupation.append(
                needsVerbose["age range"][i + j * data.occupationProb.shape[1]] + needsVerbose["Occupation"][i])
    # ageOccupation=[''.join(i) for i in zip(needsVerbose["age range"].map(str),needsVerbose["Occupation"])]
    thresh = torch.div(data.alpha_paramSchool, data.alpha_paramSchool + data.beta_paramSchool)
    if type == 'CBG based simulation':
        avgCPU = (data.BBNSch.sum(-1) / data.BBNSch.shape[1]).cpu()
        thresh = torch.mul(thresh, avgCPU)
        xLim = data.BBNSch.sum(-1) / data.BBNSch.shape[1]
        xLim = xLim.tolist()
    else:
        xLim = needsVerbose['Education services']
    thresh = thresh.tolist()

    threshStr = [str(s) for s in thresh]

    for i in range(len(needsVerbose)):
        fig.add_trace(go.Scatter(
            x=[0, xLim[i]],
            y=[ageOccupation[i], ageOccupation[i]],
            orientation='h',
            line=dict(color='rgb(244,165,130)', width=8),
        ),row=2, col=1)

    fig.add_trace(go.Scatter(
        x=thresh,
        y=ageOccupation,
        marker=dict(color='#CC5700', size=14),
        mode='markers+text',
        text=threshStr,
        textposition='middle left',
        name='alpha/(alpha+beta)'),row=2, col=1)

    fig.update_layout(title=type, showlegend=False)

    # fig.show()

    # fig = go.Figure()

    ageOccupation = []
    for i in range(data.occupationProb.shape[1]):
        for j in range(data.occupationProb.shape[0]):
            ageOccupation.append(
                needsVerbose["age range"][i + j * data.occupationProb.shape[1]] + needsVerbose["Occupation"][i])
    # ageOccupation=[''.join(i) for i in zip(needsVerbose["age range"].map(str),needsVerbose["Occupation"])]
    thresh = torch.div(data.alpha_paramReligion, data.alpha_paramReligion + data.beta_paramReligion)
    if type == 'CBG based simulation':
        avgCPU = (data.BBNRel.sum(-1) / data.BBNRel.shape[1]).cpu()
        thresh = torch.mul(thresh, avgCPU)
        xLim = data.BBNRel.sum(-1) / data.BBNRel.shape[1]
        xLim = xLim.tolist()
    else:
        xLim = needsVerbose['Religious services']
    thresh = thresh.tolist()

    threshStr = [str(s) for s in thresh]

    for i in range(len(needsVerbose)):
        fig.add_trace(go.Scatter(
            x=[0, xLim[i]],
            y=[ageOccupation[i], ageOccupation[i]],
            orientation='h',
            line=dict(color='rgb(244,165,130)', width=8),
        ),row=3, col=1)

    fig.add_trace(go.Scatter(
        x=thresh,
        y=ageOccupation,
        marker=dict(color='#CC5700', size=14),
        mode='markers+text',
        text=threshStr,
        textposition='middle left',
        name='alpha/(alpha+beta)'),row=3, col=1)

    fig.update_layout(title=type, showlegend=False)

    fig.show()


    print("!!!")
