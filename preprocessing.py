from graph import Graph
import pandas as pd

def load_and_process() -> Graph:
    # load data
    df = pd.read_csv('players_22.csv')  #dtype warning while loading data

    #get dataset for players in English premier league
    df_prem = df[df['league_name'] == 'English Premier League']

    #initially select columns which are need to create sparse network
    df_prem = df_prem[['short_name', 'player_positions', 'club_name']]
    df_prem['player_positions'] = df_prem['player_positions'].str.split(',')
    #reset index
    df_prem = df_prem.reset_index()
    #len of dataframe
    df_prem_len = df_prem.shape[0]

    #initialise graph
    g = Graph(df_prem_len)

    #w1 is the weight of the edge between two players when they play in atleast 2 same positions
    w1 = 0.66
    #w2 represents the weight of the edge between players in the same club or have 1 position which is same to another player with 1 position
    w2 = 0.33

    #iterate over dataset and create nodes
    for i in range(df_prem_len):
        for j in range(i+1, df_prem_len):
            player1 = df_prem.iloc[i]
            player2 = df_prem.iloc[j]
            player1['player_positions'] = [x.strip(' ') for x in player1['player_positions']]
            player2['player_positions'] = [x.strip(' ') for x in player2['player_positions']]
            
            if(len(player1['player_positions']) == 1 and len(player2['player_positions']) == 1):
                if(player1['player_positions'][0] == player2['player_positions'][0]):
                    g.add_edge(player1['index'], player2['index'], w2)

            if(len(set(player1['player_positions']).intersection(player2['player_positions'])) >= 2):
                 g.add_edge(player1['index'], player2['index'], w1)

            if(player1['club_name'] == player2['club_name']):
                g.add_edge(player1['index'], player2['index'], w2)

    return g
