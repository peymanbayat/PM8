import streamlit as slt
from sklearn.svm import SVC,SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import plot_confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score,mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import scipy.cluster.hierarchy as sch

def main():
	slt.title('Visualize Classification and Regression ')
	slt.subheader('Classifiers - Naive Bayes , Kernel SVM , Support Vector Machine')
	slt.subheader('Regression - Linear Regression , Polynomial Regression , Random Forest')
	slt.subheader('Clustering - K Means Clustering, Hierarchical Clustering')

	slt.sidebar.title("SELECT YOUR ALGORITHM")
	select=slt.sidebar.selectbox("Try Classification or Regression",("Classification", "Regression","Clustering"))
	if select=='Classification':
		@slt.cache(persist=True)
		def fetch_data():
			data=pd.read_csv('Social_Network_Ads.csv')
			x=data.iloc[:,[2,3]].values
			y=data.iloc[:,-1].values

			return x,y

		@slt.cache(persist=True)
		def split_data(x,y):
			x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
			sc=StandardScaler()
			x_train=sc.fit_transform(x_train)
			x_test=sc.transform(x_test)
			return x_train,x_test,y_train,y_test

		def plot_values(listofmetrics):
			if 'Confusion Matrix' in listofmetrics:
				slt.subheader('Confusion Matrix')
				plot_confusion_matrix(model,x_test,y_test,display_labels=class_names,cmap='viridis',)
				slt.pyplot()

			if 'Color Map' in listofmetrics:
				slt.subheader("Color Map - Feature Scaling has been applied")
				X_set, y_set = x_test, y_test
				X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
				                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
				plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
				             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
				plt.xlim(X1.min(), X1.max())
				plt.ylim(X2.min(), X2.max())
				for i, j in enumerate(np.unique(y_set)):
				    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				                c = ListedColormap(('darkred', 'green'))(i), label = j)
				plt.xlabel('Age')
				plt.ylabel('Estimated Salary')
				plt.legend()
				slt.pyplot()
	        

		x,y=fetch_data()
		class_names=['notpurchased','purchased']

		x_train,x_test,y_train,y_test=split_data(x,y)

		classifier = slt.sidebar.selectbox("Classifier", ("Kernel SVM","Naive Bayes","Support Vector Machine"))
		if classifier == 'Support Vector Machine':
			slt.sidebar.subheader("Model Hyperparameters")
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix','Color Map'))

			if slt.sidebar.button("Classify", key='classify'):
				slt.subheader("Support Vector Machine  Results")
				model = SVC(kernel='linear', random_state=0)
				model.fit(x_train, y_train)
				accuracy = model.score(x_test, y_test)
				y_pred = model.predict(x_test)
				slt.write("Accuracy: ", accuracy.round(2))
				slt.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
				plot_values(metrics)
				
		if classifier == 'Kernel SVM':
			slt.sidebar.subheader("Model Hyperparameters")
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix','Color Map'))
			kernel = slt.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')

			if slt.sidebar.button("Classify", key='classify'):
				slt.subheader("Kernel SVM Results")
				model = SVC(kernel=kernel, random_state=0)
				model.fit(x_train, y_train)
				accuracy = model.score(x_test, y_test)
				y_pred = model.predict(x_test)
				slt.write("Accuracy: ", accuracy.round(2))
				slt.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
				plot_values(metrics)
		if classifier == 'Naive Bayes':
			slt.sidebar.subheader("Model Hyperparameters")
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix','Color Map'))
			if slt.sidebar.button("Classify", key='classify'):
				slt.subheader("Naive Bayes Results")
				model = GaussianNB()
				model.fit(x_train, y_train)
				accuracy = model.score(x_test, y_test)
				y_pred = model.predict(x_test)
				slt.write("Accuracy: ", accuracy.round(2))
				slt.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
				plot_values(metrics)

		if slt.sidebar.checkbox("Show Dataset", False):
			slt.subheader("Classification Dataset ")
			slt.write("Customer Purchase Staus based on Social Media Ads")
			d=pd.read_csv('Social_Network_Ads.csv')
			slt.write(d)
	elif select=='Regression':
		@slt.cache(persist=True)
		def fetch_data():
			data=pd.read_csv('salary_data.csv')

			x=data.iloc[:,[0:-1]].values
			y=data.iloc[:,-1].values

			return x,y

		@slt.cache(persist=True)
		def split_data(x,y):
			x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
			sc=StandardScaler()
			x_train=sc.fit_transform(x_train)
			x_test=sc.transform(x_test)
			return x_train,x_test,y_train,y_test

		def plot_values(listofmetrics):
			if 'Graph - Train Predictions' in listofmetrics:
				slt.subheader('Graph - Train Predictions')
				plt.scatter(y_train,y_train_pred,color='red')
				plt.plot(y_train,y_train,color='blue')
				plt.title('Estimated vs Actual')
				plt.xlabel('Actual ')
				plt.ylabel('Estimated')
				slt.pyplot()
			if 'Graph - Test Predictions' in listofmetrics:
				slt.subheader('Graph - Test Predictions')
				plt.scatter(y_test,y_pred,color='red')
				plt.plot(y_test,y_test,color='blue')
				plt.title('Estimated vs Actual')
				plt.xlabel('Actual')
				plt.ylabel('Estimated')
				slt.pyplot()

		x,y=fetch_data()
		x_train,x_test,y_train,y_test=split_data(x,y)
		regressor = slt.sidebar.selectbox("Regressor", ("Linear Regression","Polynomial Regression","Random Forest Regression"))
		if regressor == 'Linear Regression':
			slt.sidebar.subheader("Model Hyperparameters")
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Graph - Train Predictions','Graph - Test Predictions'))

			if slt.sidebar.button("Predict", key='predict'):
				slt.subheader("Linear Regression  Results")
				model=LinearRegression()
				model.fit(x_train, y_train)
				accuracy = model.score(x_test, y_test)
				y_train_pred=model.predict(x_train)
				y_pred = model.predict(x_test)
				slt.write("Accuracy: ", accuracy.round(2))
				plot_values(metrics)

		if regressor == 'Polynomial Regression':
			slt.sidebar.subheader("Model Hyperparameters")
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Graph - Predictions',))
			if slt.sidebar.button("Predict", key='predict'):
				slt.subheader("Polynomial Regression Results")
				poly_reg = PolynomialFeatures(degree = 4)
				x_poly = poly_reg.fit_transform(x)
				poly_reg.fit(x_poly, y)
				model = LinearRegression()
				model.fit(x_poly, y)
				accuracy = model.score(poly_reg.fit_transform(x), y)
				slt.write("Accuracy: ", accuracy.round(2))
				plt.scatter(x,y,color='red')
				plt.plot(x,model.predict(poly_reg.fit_transform(x)))
				plt.title('Experience Vs Salary')
				plt.xlabel('Experience')
				plt.ylabel('Salary')
				slt.pyplot()
				
		if regressor == 'Random Forest Regression':
			slt.sidebar.subheader("Model Hyperparameters")
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Graph - Predictions',))
			if slt.sidebar.button("Predict", key='predict'):
				slt.subheader("Random Forest Regression Results")
				model = RandomForestRegressor(n_estimators = 10, random_state = 0)
				model.fit(x_train,y_train)
				accuracy = model.score(x_test, y_test)
				slt.write("Accuracy: ", accuracy.round(2))
				X_grid = np.arange(min(x_train), max(x_train), 0.01)
				X_grid = X_grid.reshape((len(X_grid), 1))
				plt.scatter(x_train, y_train, color = 'red')
				plt.plot(X_grid, model.predict(X_grid), color = 'blue')
				plt.title('Experience Vs Salary')
				plt.xlabel('Experience')
				plt.ylabel('Salary')
				slt.pyplot()

		if slt.sidebar.checkbox("Show Dataset", False):
			slt.subheader("Regression Dataset ")
			slt.write("Experience vs Salary Dataset")
			d=pd.read_csv('salary_data.csv')
			slt.write(d)
	else:
		@slt.cache(persist=True)
		def fetch_data():
			data=pd.read_csv('Mall_Customers.csv')
			x = data.iloc[:, [3, 4]].values
			return x
		def plot_values(listofmetrics):
			if 'Color Map' in listofmetrics:
				colors=['red','blue','green','cyan','magenta','sienna','lightpink','black','chocalate','violet']
				for i in range(n_clusters):
					plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = colors[i], label = 'Cluster '+str(i+1))
				if centroid=='kmeans':
					plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')
				plt.title('Clusters of customers')
				plt.xlabel('Annual Income ')
				plt.ylabel('Spending Score (1-100)')
				plt.legend()
				slt.pyplot()

		X=fetch_data()
		cluster = slt.sidebar.selectbox("Cluster", ('K Means Clustering','Hierarchical Clustering'))
		if cluster=='K Means Clustering':
			slt.sidebar.subheader("Use elbow method to find the optimal nnumber of clusters")
			if slt.sidebar.button("Elbow Method"):
				wcss=[]
				for i in range(1,11):
					kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10)
					kmeans.fit(X)
					wcss.append(kmeans.inertia_)
				plt.plot(range(1,11),wcss)
				plt.title('The Elbow Method')
				plt.xlabel('No of Clusters')
				plt.ylabel('wcss')
				slt.pyplot()
			slt.sidebar.subheader("Model Hyperparameters")
			n_clusters = slt.sidebar.number_input("Choose the number of clusters", 1, 8, step=1, key='noofclusters')
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Color Map',))
			if slt.sidebar.button("Cluster", key='cluster'):
				slt.subheader("K means Clustering Results")
				kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 42)
				y_kmeans = kmeans.fit_predict(X)
				centroid='kmeans'
				plot_values(metrics)
		else:
			slt.sidebar.subheader("Use Dendrogram to find the optimal number of clusters")
			if slt.sidebar.button('Dendrogram'):
				dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
				plt.title('Dendrogram')
				plt.xlabel('Customers')
				plt.ylabel('distance (euclidean')
				slt.pyplot()
			slt.sidebar.subheader("Model Hyperparameters")
			n_clusters = slt.sidebar.number_input("Choose the number of clusters", 1, 8, step=1, key='noofclusters')
			metrics = slt.sidebar.multiselect("What metrics to plot?", ('Color Map',))
			if slt.sidebar.button("Cluster", key='cluster'):
				slt.subheader("Hierarchical Clustering Results")
				model = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'euclidean', linkage='ward')
				y_kmeans = model.fit_predict(X)
				centroid='hierarchy'
				plot_values(metrics)

		if slt.sidebar.checkbox("Show Dataset", False):
			slt.subheader("Clustering Dataset ")
			slt.write("Annual vs Spending Score")
			d=pd.read_csv('Mall_Customers.csv')
			slt.write(d)





if __name__ == '__main__':
    main()
