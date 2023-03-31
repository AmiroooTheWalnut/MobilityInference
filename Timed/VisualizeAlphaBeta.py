
import pandas as pd
import numpy as np
import io
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import torch
import torch.distributions.constraints as constraints
import pyro

def showAlphaBetaRange(type,alpha,beta,needsVerbose):
    fig = go.Figure()

    ageOccupation=[''.join(i) for i in zip(needsVerbose["age range"].map(str),needsVerbose["Occupation"])]
    thresh=torch.div(alpha, alpha + beta)
    threshStr = [str(s) for s in thresh.tolist()]

    for i in range(len(needsVerbose)):
        fig.add_trace(go.Scatter(
            x=[0, needsVerbose['Retailing services'][i]],
            y=[ageOccupation[i], ageOccupation[i]],
            orientation='h',
            line=dict(color='rgb(244,165,130)', width=8),
        ))

    fig.add_trace(go.Scatter(
        x=thresh,
        y=ageOccupation,
        marker=dict(color='#CC5700', size=14),
        mode='markers+text',
        text=threshStr,
        textposition='middle left',
        name='Woman'))

    fig.update_layout(title="Sample plotly", showlegend=False)

    fig.show()

    print("!!!")

    # data = '''
    #  Grade Women Men
    # 0 "Less Than 9th Grade" 21 34
    # 1 "9th 12th(no degree)" 22 37
    # 2 "Hight School" 29 47
    # 3 "Some college, no degree" 35 53
    # 4 "Associate Degree" 38 56
    # 5 "Bachelor's Degree" 54 84
    # 6 "Master's Degree" 69 99
    # 7 "Doctorate Degree" 91 151
    # '''
    #
    # df = pd.read_csv(io.StringIO(data), sep='\s+')
    # df.sort_values('Men', ascending=False, inplace=True, ignore_index=True)
    #
    # w_lbl = [str(s) for s in df['Women'].tolist()]
    # m_lbl = [str(s) for s in df['Men'].tolist()]
    #
    # fig = go.Figure()
    #
    # for i in range(0, 8):
    #     fig.add_trace(go.Scatter(
    #         x=[df['Women'][i], df['Men'][i]],
    #         y=[df['Grade'][i], df['Grade'][i]],
    #         orientation='h',
    #         line=dict(color='rgb(244,165,130)', width=8),
    #     ))
    #
    # fig.add_trace(go.Scatter(
    #     x=df['Women'],
    #     y=df['Grade'],
    #     marker=dict(color='#CC5700', size=14),
    #     mode='markers+text',
    #     text=w_lbl,
    #     textposition='middle left',
    #     name='Woman'))
    #
    # fig.add_trace(go.Scatter(
    #     x=df['Men'],
    #     y=df['Grade'],
    #     marker=dict(color='#227266', size=14),
    #     mode='markers+text',
    #     text=m_lbl,
    #     textposition='middle right',
    #     name='Men'))
    #
    # fig.update_layout(title="Sample plotly", showlegend=False)
    #
    # fig.show()