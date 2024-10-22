import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

file_paths = ['clusters-4-v0.csv','clusters-4-v1.csv','clusters-4-v2.csv']
models = {
    "Logistic Regression":LogisticRegression(),
    "SVC (linear)" :SVC(kernel='linear',probability=True), 
    "SVC (rbf)":SVC (kernel='rbf',probability=True),
    "Random Forest Classifier (1)":RandomForestClassifier(min_samples_leaf=1,),
    "Random Forest Classifier (3)":RandomForestClassifier(min_samples_leaf=3),
    "Random Forest Classifier (5)":RandomForestClassifier(min_samples_leaf=5),
    "Neural Network Classifier (5)":MLPClassifier(hidden_layer_sizes=(5)),
    "Neural Network Classifier (5,5)":MLPClassifier(hidden_layer_sizes=(5,5)),
    "Neural Network Classifier (5,5,5)":MLPClassifier(hidden_layer_sizes=(5,5,5)),
    "Neural Network Classifier (10)":MLPClassifier(hidden_layer_sizes=(10))
    }

# Create Pair Plots for the given datasets to visualize the data distribution
def create_pairplots(file_paths):
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        sns.pairplot(data, hue='y', palette='viridis')
        plt.suptitle(f'Pair Plot for {file_path}')
        plt.show()

create_pairplots(file_paths)

# evaluate the model using the given metrics - accuracy, precision, recall, f1, auc
def evaluate_model(y_test, y_predict, y_prob):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_predict)
    
    precision_per_class = precision_score(y_test, y_predict, average=None)
    for i in range(4):
        metrics[f'precision_{i+1}'] = precision_per_class[i]
    
    metrics['precision_avg'] = precision_score(y_test, y_predict, average='macro')
    
    recall_per_class = recall_score(y_test, y_predict, average=None)
    for i in range(4):
        metrics[f'recall_{i+1}'] = recall_per_class[i]
    
    metrics['recall_avg'] = recall_score(y_test, y_predict, average='macro')
    
    f1_per_class = f1_score(y_test, y_predict, average=None)
    for i in range(4):
        metrics[f'f1_{i+1}'] = f1_per_class[i]
    
    metrics['f1_avg'] = f1_score(y_test, y_predict, average='macro')
    
    auc_per_class = roc_auc_score(y_test, y_prob, average=None, multi_class='ovr')
    for i in range(4):
        metrics[f'auc_{i+1}'] = auc_per_class[i]
    
    metrics['auc_avg'] = roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr')
    
    return metrics

# Plot the decision boundaries for the given model
def plot_decision_boundaries(ax,model,model_name, X, y):
    x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
    y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    ax.scatter(X['x1'], X['x2'], c=y, cmap='viridis', edgecolor='k')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(model_name)

# Plot the ROC curve for the given model
def plot_roc(ax,y_true, y_prob, model_name,dataset):
    
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} - {dataset} Data')
    ax.legend(loc="lower right")

# Process the data using the given models and store the results in a dataframe
def process_data(file_paths,models):
    results = []
    
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        X = data[['x1','x2']]
        y = data['y']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name,model in models.items():
            model.fit(X_train,y_train)
            
            y_train_predict = model.predict(X_train)
            y_train_prob = model.predict_proba(X_train)
            metrics_train = evaluate_model(y_train, y_train_predict, y_train_prob)
            results.append({
                'file':file_path,
                'model':model_name,
                'dataset':'train',
                **metrics_train
            })
            
            y_test_predict = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)
            
            metrics_test = evaluate_model(y_test, y_test_predict, y_test_prob)
            
            results.append({
                'file':file_path,
                'model':model_name,
                'dataset':'test',
                **metrics_test
            })
            
            fig,axs = plt.subplots(1,3,figsize=(18,6))
            
            plot_decision_boundaries(axs[0],model,model_name, X_test, y_test)
            plot_roc(axs[1],y_train, y_train_prob, model_name,'Train')
            plot_roc(axs[2],y_test, y_test_prob, model_name,'Test') 
            
            plt.suptitle(f'{model_name} - {file_path}')
            plt.show()
            
    return results
            

results = process_data(file_paths,models)

results_df = pd.DataFrame(results)
results_df.to_csv('results.csv',index=False)
results_df