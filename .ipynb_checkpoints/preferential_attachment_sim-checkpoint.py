import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#import networkx as nx
import plotly.express as px


def init_web(authors):
    # columns of the dataframe we init later
    cols = ['citing_author', 'paper_citing_id', 'cited_author', 'paper_cited_id', 'time']
    # hold the data we create later
    data = []
    # for all of the authors that is passed into the dataframe, have them cite another outher
    for citing_author in authors:
        # have a random citation between authors
        cited_author = random.choice(authors)
        # no author should cite themselves on the first step
        while cited_author == citing_author:
            cited_author = random.choice(authors)
        # append the citations into the data with the paper indexes and time step
        data.append([citing_author, authors.index(citing_author), cited_author, authors.index(cited_author), 0])
        # create and return a dataframe of the data
    return pd.DataFrame(data, columns=cols)


def get_in_degrees(df):
    # probability function of a new edge connecting to any node with a degree k is
    in_degrees = df[['cited_author', 'paper_cited_id']].value_counts()

    # include the authors who didn't get cited
    for row in df[['citing_author', 'paper_citing_id']].values.tolist():
        author_paper = tuple(row)
        if author_paper not in in_degrees.index:
            zero_citations = pd.Series({author_paper: 0})
            in_degrees = pd.concat([in_degrees, zero_citations])
    return in_degrees


def get_p_k(df, k_0=1):
    # probability function of a new edge connecting to any node with a degree k is
    in_degrees = df[['cited_author', 'paper_cited_id']].value_counts()

    # include the authors who didn't get cited
    for row in df[['citing_author', 'paper_citing_id']].values.tolist():
        author_paper = tuple(row)
        if author_paper not in in_degrees.index:
            zero_citations = pd.Series({author_paper: 0})
            in_degrees = pd.concat([in_degrees, zero_citations])

    # count the degrees that are present in our dataset
    degree_counts = in_degrees.value_counts()

    # using Price's model for publishing calculate the probability that a new author will cite a paper of a certain degree
    p_k = degree_counts / degree_counts.sum()

    return p_k


def get_citation(df, k_0=1):
    p_k = get_p_k(df, k_0=1)
    # probability function of a new edge connecting to any node with a degree k is
    in_degrees = df[['cited_author', 'paper_cited_id']].value_counts()

    # include the authors who didn't get cited
    for row in df[['citing_author', 'paper_citing_id']].values.tolist():
        author_paper = tuple(row)
        if author_paper not in in_degrees.index:
            zero_citations = pd.Series({author_paper: 0})
            in_degrees = pd.concat([in_degrees, zero_citations])

    # count the degrees that are present in our dataset
    degree_counts = in_degrees.value_counts()

    # using Price's model for publishing calculate the probability that a new author will cite a paper of a certain degree
    p_k = degree_counts / degree_counts.sum()
    k_probs = ((p_k.index + k_0) * p_k) / sum((p_k.index + k_0) * p_k)

    # chose a random degree with the weight from the Price Model
    random_degree = random.choices(k_probs.index, weights=k_probs, k=1)[0]

    # from the chose degree, find an author with that many citations. Sometimes there are several, so we randomly
    # select from that subset
    chosen = in_degrees[in_degrees == random_degree]

    # randomly chose and other that has the specified degree
    num_chosen = chosen.shape[0]
    if num_chosen == 1:
        newly_cite = chosen.index.values[0]
    else:
        newly_cite = random.choice(chosen.index.values)

    return newly_cite


def step(df, authors, cols=['citing_author', 'paper_citing_id', 'cited_author', 'paper_cited_id', 'time']):
    next_step = []
    current_paper_ix = df['time'].max()
    for i, auth in enumerate(authors):
        cited = get_citation(df)
        citing_id = df['paper_citing_id'].max() + i + 1
        next_step.append([auth, citing_id, cited[0], cited[1], current_paper_ix + 1])
    next_step = pd.DataFrame(next_step, columns=cols)
    return pd.concat([df, next_step], axis=0, ignore_index=True)


def step_multiple_cite(df, authors, max_cite=1,
                       cols=['citing_author', 'paper_citing_id', 'cited_author', 'paper_cited_id', 'time']):
    # max_cite must be less than the current papers that exist!
    next_step = []
    current_paper_ix = df['time'].max()
    # for each author figure out who they should cite
    for i, auth in enumerate(authors):

        citing_id = df['paper_citing_id'].max() + i + 1
        # for a random number of citations limited by the max number of cites, pull a random citation and create the meta data
        # range starts at 0, and doesn't include the stop number
        for cite_num in range(1, random.randint(1, max_cite) + 1):
            # pull a citation according to price's model
            cited = get_citation(df)
            while [auth, citing_id, cited[0], cited[1], current_paper_ix + 1] in next_step:
                cited = get_citation(df)

            next_step.append([auth, citing_id, cited[0], cited[1], current_paper_ix + 1])
    next_step = pd.DataFrame(next_step, columns=cols)
    return pd.concat([df, next_step], axis=0, ignore_index=True)


def take_steps(df, authors, num_of_steps):
    for i in range(num_of_steps):
        # df = step_multiple_cite(df, authors, max_cite)
        df = step(df, authors)
    return df


def take_steps_multiple_cite(df, authors, num_of_steps, max_cite=1):
    if max_cite > len(authors) - 1:
        print(
            'max_cite cannot be larger than the number of papers, which is the case if max_cite > len(authors) - 1 at the start')
    for i in range(num_of_steps):
        df = step_multiple_cite(df, authors, max_cite)
        # df = step(df, authors)
    return df


def get_return(df):
    return df['cited_author'].value_counts() / df['cited_author'].value_counts().sum()