#credit: https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson,skellam
from scipy.optimize import minimize, fmin
from multiprocessing import Pool
from pathlib import Path

def calc_means(param_dict, homeTeam, awayTeam):
    return [np.exp(param_dict['attack_'+homeTeam] + param_dict['defence_'+awayTeam] + param_dict['home_adv']),
            np.exp(param_dict['defence_'+homeTeam] + param_dict['attack_'+awayTeam])]

def rho_correction(x, y, lambda_x, mu_y, rho):
    if x==0 and y==0:
        return 1- (lambda_x * mu_y * rho)
    elif x==0 and y==1:
        return 1 + (lambda_x * rho)
    elif x==1 and y==0:
        return 1 + (mu_y * rho)
    elif x==1 and y==1:
        return 1 - rho
    else:
        return 1.0

def solve_parameters(dataset, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                     constraints=None, **kwargs):
    teams = np.sort(dataset['HomeTeam'].unique())
    # check for no weirdness in dataset
    away_teams = np.sort(dataset['AwayTeam'].unique())
    if not np.array_equal(teams, away_teams):
        raise ValueError("Something's not right")
    n_teams = len(teams)
    if init_vals is None:
        # random initialisation of model parameters
        init_vals = np.concatenate((np.random.uniform(0,1,(n_teams)), # attack strength
                                      np.random.uniform(0,-1,(n_teams)), # defence strength
                                      np.array([0, 1.0]) # rho (score correction), gamma (home advantage)
                                     ))
    if constraints is None:
        # Summe der Attack-Staerken auf n_teams fixieren (statt hardcoded 20 aus dem
        # Premier-League-Original) -- sonst ist die Nebenbedingung bei abweichender
        # Teamzahl (D1+D2 kombiniert, mehrere Saisons) einfach falsch.
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:n_teams]) - n_teams}]

    def dc_log_like(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma):
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x) 
        return (np.log(rho_correction(x, y, lambda_x, mu_y, rho)) + 
                np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))

    def estimate_paramters(params):
        score_coefs = dict(zip(teams, params[:n_teams]))
        defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
        rho, gamma = params[-2:]
        log_like = [dc_log_like(row.HomeGoals, row.AwayGoals, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                     score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam], rho, gamma) for row in dataset.itertuples()]
        return -sum(log_like)
    opt_output = minimize(estimate_paramters, init_vals, options=options, constraints = constraints, **kwargs)
    if debug:
        # sort of hacky way to investigate the output of the optimisation process
        return opt_output
    else:
        return dict(zip(["attack_"+team for team in teams] + 
                        ["defence_"+team for team in teams] +
                        ['rho', 'home_adv'],
                        opt_output.x)) 

def dixon_coles_simulate_match(params_dict, homeTeam, awayTeam, max_goals=10):
    team_avgs = calc_means(params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                   team_avgs[1], params_dict['rho']) for away_goals in range(2)]
                                   for home_goals in range(2)])
    output_matrix[:2,:2] = output_matrix[:2,:2] * correction_matrix
    return output_matrix

def solve_parameters_decay(dataset, xi=0.001, debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                     constraints=None, **kwargs):
    teams = np.sort(dataset['HomeTeam'].unique())
    # check for no weirdness in dataset
    away_teams = np.sort(dataset['AwayTeam'].unique())
    if not np.array_equal(teams, away_teams):
        raise ValueError("something not right")
    n_teams = len(teams)
    if init_vals is None:
        # random initialisation of model parameters
        init_vals = np.concatenate((np.random.uniform(0,1,(n_teams)), # attack strength
                                      np.random.uniform(0,-1,(n_teams)), # defence strength
                                      np.array([0,1.0]) # rho (score correction), gamma (home advantage)
                                     ))
    if constraints is None:
        # gleiche Begruendung wie in solve_parameters: dynamisch statt hardcoded 20
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:n_teams]) - n_teams}]

    def dc_log_like_decay(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma, t, xi=xi):
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x) 
        return  np.exp(-xi*t) * (np.log(rho_correction(x, y, lambda_x, mu_y, rho)) + 
                                  np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))

    def estimate_paramters(params):
        score_coefs = dict(zip(teams, params[:n_teams]))
        defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
        rho, gamma = params[-2:]
        log_like = [dc_log_like_decay(row.HomeGoals, row.AwayGoals, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                                      score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam], 
                                      rho, gamma, row.time_diff, xi=xi) for row in dataset.itertuples()]
        return -sum(log_like)
    opt_output = minimize(estimate_paramters, init_vals, options=options, constraints = constraints, **kwargs)
    if debug:
        # sort of hacky way to investigate the output of the optimisation process
        return opt_output
    else:
        return dict(zip(["attack_"+team for team in teams] + 
                        ["defence_"+team for team in teams] +
                        ['rho', 'home_adv'],
                        opt_output.x))

def get_1x2_probs(match_score_matrix):
    return dict({"H":np.sum(np.tril(match_score_matrix, -1)), 
                 "A":np.sum(np.triu(match_score_matrix, 1)), "D":np.sum(np.diag(match_score_matrix))})

def build_temp_model(dataset, time_diff, xi=0.000, init_params=None):
    test_dataset = dataset[((dataset['time_diff']<=time_diff) & (dataset['time_diff']>=time_diff-2))]
    if len(test_dataset)==0:
        return 0
    train_dataset = dataset[dataset['time_diff']>time_diff]
    train_dataset['time_diff'] = train_dataset['time_diff'] - time_diff
    params = solve_parameters_decay(train_dataset, xi=xi, init_vals=init_params)
    predictive_score = sum([np.log(get_1x2_probs(dixon_coles_simulate_match(
                    params, row.HomeTeam, row.AwayTeam))[row.FTR]) for row in test_dataset.itertuples()])
    return predictive_score    
    
def check_xi(match_day):
    xi_score = build_temp_model(data, match_day, xi=my_xi)
    return xi_score

def get_data():
    d1 = pd.read_csv(DATA_DIR / "d1_clean.csv")[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Div']].rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
    d2 = pd.read_csv(DATA_DIR / "d2_clean.csv")[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Div']].rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

    d1['Date'] = pd.to_datetime(d1['Date'],  format='mixed')
    d2['Date'] = pd.to_datetime(d2['Date'],  format='mixed')

    # Erst concaten, DANN time_diff aus einem gemeinsamen globalen Referenzdatum
    # berechnen -- sonst ist time_diff zwischen D1 und D2 nicht vergleichbar, falls
    # beide Ligen nicht exakt am selben Tag zuletzt gespielt haben.
    combined = pd.concat([d1, d2], ignore_index=True).sort_values('Date').reset_index(drop=True)
    combined['time_diff'] = (combined['Date'].max() - combined['Date']).dt.days

    return combined


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
my_xi = 0.00325


if __name__ == "__main__":
    data = get_data()
    match_days = [day for day in range(99,-1,-3) if len(data[((data['time_diff']<=day) & (data['time_diff']>=(day-2)))])]
    print("Pooling...")
    with Pool() as pool:         # start worker processes (number will depend on your computer's architecture)
        xi_result = pool.map(check_xi, match_days)
    with open('find_xi_5season_{}.txt'.format(str(my_xi)[2:]), 'wb') as thefile:
        pickle.dump(xi_result, thefile)