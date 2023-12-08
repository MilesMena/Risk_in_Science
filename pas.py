import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def init_web(num_authors, seed = None):
    # create data array[array[int]] where arrays of data are citing_author_id, citing_paper_id. 
    # later we concatenate cited_author_id, and cited_paper_id, and time
    data = np.array([np.arange(num_authors), np.arange(num_authors)])
    # create an array of what author is getting cited
    if seed != None:
        np.random.seed(seed)
    # name the array that we are using so that we don't have to call it so many times
    author_ids = data[0,:]
    #print(len(author_ids))
    cited_authors = np.random.choice(author_ids, num_authors, replace=True)
    # authors cannot cite themselves on the first paper
    while any(np.where(cited_authors == author_ids, True, False)):
        cited_authors = np.random.choice(author_ids, num_authors, replace=True)
    # index of where the cited author is in the prior dataset. 
    # not really necassary since right now the index of the author id is the author's id 
    # but helpful for later, when the author has multiple papers out
    cited_authors_index = [np.where(author_ids == cited_author)[0][0] for cited_author in cited_authors]
    # from the array that contains papers, take the papers with the same index as the authors that were cited
    cited_paper_ids = data[1,:][cited_authors_index]
    round_of_papers = np.array([cited_authors,cited_paper_ids])
    return np.concatenate([data,round_of_papers, [np.zeros(data.shape[1])]])

def get_in_degrees(data):
    citing_papers = np.unique(data[1,:])# , return_counts=True)
    #print(citing_papers)
    return dict((x,list(data[3,:]).count(x)) for x in citing_papers)

def get_degree_counts(in_degrees):
    return dict((degree,list(in_degrees.values()).count(degree)) for degree in in_degrees.keys())

def get_p_k(degree_counts):
    total_in_degrees = sum(degree_counts.values())
    return dict((key,degree_counts[key] / total_in_degrees) for key in degree_counts)
def get_k_probs(p_k, k_0 = 1):
    p_k_degrees  = np.array(list(p_k.keys())) 
    p_k_vals  = np.array(list(p_k.values())) 
    return ((p_k_degrees + k_0) * p_k_vals) / sum((p_k_degrees + k_0) * p_k_vals)
def get_citations(data):
    in_degrees = get_in_degrees(data)
    degree_counts = get_degree_counts(in_degrees)
    p_k = get_p_k(degree_counts)
    k_probs = get_k_probs(p_k)
    
    # for every unqiue author in our data, tell them which degre of a paper they should cite
    
    unique = np.unique(data[0,:])
    #print(counts)
    random_degrees = np.random.choice(list(degree_counts.keys()),len(unique) , replace=True, p=k_probs)
    
    in_degrees = np.array(list(in_degrees.items()))
    
    random_papers = []
    for degree in random_degrees:
        paper_id_in_degree_index = np.where(in_degrees[:,1] == degree)
        papers_to_cite = in_degrees[:,0][paper_id_in_degree_index]
        random_papers.append(np.random.choice(papers_to_cite))
    return random_papers

def get_citations_risk(data, risk):
    in_degrees = get_in_degrees(data)
    degree_counts = get_degree_counts(in_degrees)
    p_k = get_p_k(degree_counts)
    k_probs = get_k_probs(p_k)
    
    # for every unqiue author in our data, tell them which degre of a paper they should cite
    prob_publish = np.array([[np.random.uniform(0,1) for i in range(num_authors)]])
    #print(prob_publish)
    unique = np.where(risk > prob_publish)[1]
    #unique = np.unique(data[0,:])
    #print(counts)
    random_degrees = np.random.choice(list(degree_counts.keys()),len(unique) , replace=True, p=k_probs)
    
    in_degrees = np.array(list(in_degrees.items()))
    
    random_papers = []
    for degree in random_degrees:
        paper_id_in_degree_index = np.where(in_degrees[:,1] == degree)
        papers_to_cite = in_degrees[:,0][paper_id_in_degree_index]
        random_papers.append(np.random.choice(papers_to_cite))
    return random_papers, unique
np.random.uniform(0,1)


def get_new_paper_ids(data, citations):
    num_papers = data[1,:].max() + 1
    return np.arange(num_papers, num_papers + len(citations))

def get_new_cited_authors(data, citations):
    
    # CAUTION. once we have multiple citations then the paper_id shows up multiple times in papers[1,:]
    # i think this will be fine since we would only be grabbing the first index where the paper is cited and that corresponds to the same author regardless

    return data[0,:][[int(c) for c in citations]]


def take_steps(s, steps,num_authors):
    for i in range(1, steps + 1):
        new_citations = get_citations(s)
        
        new_cited_authors = get_new_cited_authors(s,new_citations)
        published_paper_ids = get_new_paper_ids(s, new_citations)
        #print([np.arange(num_authors), published_paper_ids, new_cited_authors, new_citations,np.ones(len(new_citations)) * i])
        single_step = np.array([np.arange(num_authors), published_paper_ids, new_cited_authors, new_citations,np.ones(len(new_citations)) * i])
        s= np.concatenate([s,single_step], axis = 1)
    return pd.DataFrame(s.T, columns = ['citing_author','paper_citing_id','cited_author' ,'paper_cited_id', 'time' ] )



def plot_cumsum(data,unique_items, var = 'cited_author', legend = True):
    for item in unique_items:
        plt.plot(data[data[var] == item]['time'].value_counts(sort = False).cumsum())
    if legend:
        plt.legend(unique_items)
    plt.title('%s: Cumulative Sum of Citations'%var);
    plt.xlabel('Time', fontsize=12);
    plt.ylabel('Citations', fontsize=12);
    plt.show()

def plot_citations(data,unique_items, var = 'cited_author', legend = True):
    for item in unique_items:
        plt.plot(data[data[var] == item]['time'].value_counts(sort = False))
    if legend:
        plt.legend(unique_items)
    plt.title('%s: Citations Over Time'%var);
    plt.xlabel('Time', fontsize=12);
    plt.ylabel('Citations', fontsize=12);
    plt.show()

# do we have to unscale the returns now that we have them?
def get_r_var_mean(returns):
    r_bar = returns.mean(axis= 1)#.round(4).mean()
    r_s = returns.var(axis = 1)
    return r_bar, r_s

# return is the number of citations over the average numcer of citaions for that year
def get_returns(df, num_authors, cumulative_time = True):
    r_list = []
    zeros = pd.Series(np.zeros(num_authors),np.arange(num_authors), name = 'zero')
    times = []
    for t in df['time'].unique():
        #num_citation_t = df[df['time'] == t]['cited_author'].value_counts()
        if cumulative_time:
            #print(times)
            times.append(t)
            counts = df[df['time'].isin(times)]['cited_author'].value_counts().sort_index()
        else:
            counts = df[df['time'] == t]['cited_author'].value_counts().sort_index()
        avg, max, min, std = np.mean(counts), np.max(counts), np.min(counts), np.std(counts)
        #print(avg, max, min, std)
        zeros_counts = pd.merge(zeros,counts,how = 'outer', left_index = True, right_index = True).fillna(0)
        
        num_cite_t = zeros_counts['zero'] + zeros_counts['count']
        # since we always have 100 citations then there will be an average of 1
        #avg = np.mean(num_citation_t)
        # this one does NOT work
        #r = (num_citation_t - avg) / num_citation_t.max()
        
        # https://medium.com/analytics-vidhya/feature-scaling-normalization-standardization-and-scaling-c920ed3637e7
        # min max scaling. Notice that we are using the min and max from counts. Otherwise the min is 0, and that doesn't scale the data
        # so we scale by the smallest none zero number
        if max == min:
            r = 0
        else:
            r = (num_cite_t - min) / (max - min)

        
        # mean normilzation. this works well, but the numbers are super small. 
        #r = (num_cite_t - avg) / (max - min)
        # Standardization. This isn't bad but it has more positive returns than negative returns. 
        #r = (num_cite_t - avg) / num_cite_t.std()
        
        r_list.append(r)
    return pd.concat(r_list, axis = 1)

#s_df