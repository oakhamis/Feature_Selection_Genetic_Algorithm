## Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef
from genetic_selection import GeneticSelectionCV
import matplotlib.pyplot as plt
import itertools
import warnings

# Suppressing warnings for better readability
warnings.filterwarnings("ignore")

# Setting MCC as the scoring function
mcc = make_scorer(matthews_corrcoef)

## Data Preprocessing
data = pd.read_csv('bank-full.csv')
data_col = data.dtypes.pipe(lambda x: x[x == 'object']).index
label_mapping = {}
for c in data_col:
    data[c], label_mapping[c] = pd.factorize(data[c])
print(label_mapping)
x = data.drop('y', axis=1)
y = data['y']
allfeats = x.columns.tolist()
print(allfeats)

# Encoding categorical columns
data_col = data.select_dtypes(include=['object']).columns
label_mapping = {c: dict(enumerate(data[c].astype('category').cat.categories)) for c in data_col}
for c, mapping in label_mapping.items():
    data[c] = data[c].astype('category').cat.codes

x = data.drop('y', axis=1)
y = data['y']

## Decision Tree
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
clf = tree.DecisionTreeClassifier().fit(x_train, y_train)
tree.plot_tree(clf, max_depth=1)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred, normalize=False) / x_test.shape[0]

## GA Parameters & Configurations
estimator = tree.DecisionTreeClassifier()
report = pd.DataFrame(columns=['No of Feats', 'Chosen Feats', 'Scores'])

rkf = RepeatedStratifiedKFold(n_repeats = 30, n_splits = 3)
pop_size =[50]
cross_over=[0.2,0.5,0.8]
mutation = [0.01,0.05,0.1]
variations = [i for  i in itertools.product(pop_size,cross_over,mutation)]
run = 0 
best_fitness_values = [0]*len(variations)

# Initialize the lists here
nofeats = [] 
chosen_feats = [] 
cvscore = []

for var_index ,var in enumerate(variations):
  bsf_score_run = 0
  selector = GeneticSelectionCV(estimator,
                                  cv = rkf,
                                  verbose = 0,
                                  scoring = "accuracy",
                                  max_features = len(allfeats),
                                  n_population = var[0],
                                  crossover_proba = var[1],
                                  mutation_proba = var[2],
                                  n_generations = 30,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.1,
                                  #tournament_size = 3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
  for i in range(30):
    print("-------------------------run {} ----------------------".format(i))
    
    selector  = selector.fit(x_train, y_train)
    run+=1
    genfeats = data[allfeats].columns[selector.support_]
    genfeats = list(genfeats)
    print("Chosen Feats:  ", genfeats)

    cv_score = selector.generation_scores_[-1]
    if cv_score > bsf_score_run:
      bsf_score_run = cv_score
      bsf_score_index = run
      best_fitness_values[var_index] = selector.generation_scores_
      
    nofeats.append(len(genfeats)) 
    chosen_feats.append(genfeats) 
    cvscore.append(cv_score)


# Storing Results Into Dataframe & CSV file
report["No of Feats"] = nofeats
report["Chosen Feats"] = chosen_feats
report["Scores"] = cvscore


report_final = report.iloc[0:270].copy()
report_final["Scores"] = np.round(report_final["Scores"], 6)
report_final.sort_values(by = "Scores", ascending = False, inplace = True)

#report.index
ga_feats = report_final.iloc[0]["Chosen Feats"]
print(ga_feats)

report_final.to_csv("GA_report.csv",index=False)

max_fitness_value = [max(i) for i in best_fitness_values[0:9]]

index = np.argsort(max_fitness_value) 

index = index[::-1]


## GA Variation Plots

vars_ = ['Pop. Size '+ str(vara[0]) + ' / Crossover rate '+ str(vara[1]) + ' / Mutation rate ' + str(vara[2]) for vara in variations]
for i in index:
    plt.plot(best_fitness_values[i], label=vars_[i])
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1))
plt.ylim((0.883,0.8936))
plt.rcParams["figure.figsize"] = (6,6)




## Error Plots

# Standard error calculation for each variation:
First_std_err = np.std(report.iloc[30:60]['Scores']) / np.sqrt(len(report.iloc[30:60]))
Second_std_err = np.std(report.iloc[150:180]['Scores']) / np.sqrt(len(report.iloc[150:180]))   
Third_std_err = np.std(report.iloc[240:270]['Scores']) / np.sqrt(len(report.iloc[240:270]))  
   
   
# Error Plot/Figure 3 in the report.
plt.figure(figsize=(5,5))
generation = [x for x in range(len(report.iloc[240:270]))]
fig, ax = plt.subplots()
ax.errorbar(generation, report.iloc[30:60]['Scores'],yerr = First_std_err, solid_capstyle='projecting', capsize=2,label='Population size:50, Crossover Rate: 0.8, Mutation Rate:0.1')
ax.errorbar(generation, report.iloc[150:180]['Scores'],yerr = Second_std_err, solid_capstyle='projecting', capsize=2,label='Population size:50, Crossover Rate: 0.5, Mutation Rate:0.1')
ax.errorbar(generation, report.iloc[240:270]['Scores'],yerr = Third_std_err, solid_capstyle='projecting', capsize=2,label='Population size:50, Crossover Rate: 0.2, Mutation Rate:0.05')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1))
ax.grid(alpha=0.5, linestyle=':')   
plt.show()
