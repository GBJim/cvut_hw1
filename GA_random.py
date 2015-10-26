import pandas as pd
import numpy as np
import random 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer




class RandomVectors(object):

	def __init__(self, X,Y):

		vectors_shape = (X.shape[0], X.shape[1] +1)
		self.X = np.ones(vectors_shape ,dtype=float)
		self.X[:,:-1] = X
		self.Y = Y
		self.vectors = np.random.rand(vectors_shape[1])

	def predict(self):
		return np.sum(self.vectors * self.X, axis=1) - self.Y

	def error(self):
		prediction = self.predict()
		return mean_squared_error(Y, prediction)

	def random_search(self,n_interations=100):
		best_vectors = self.vectors
		min_error = self.error()

		for i in range(n_interations):
			self.vectors = np.random.rand(self.vectors.shape[0])
			error = self.error()
			if error < min_error:
				error = min_error
				best_vectors = self.vectors

		self.vectors = best_vectors


class GeneVector(RandomVectors):

	def __init__(self, X,Y, vectors = None):

		if vectors is None:
			super().__init__(X,Y)

		else:
			self.X = X
			self.Y = Y
			self.vectors = vectors

	def mutate(self):
		mutation_index = random.randint(0, self.vectors.shape[0]-1)
		self.vectors[mutation_index] = random.random()

	def crossover(self, mate):
		exchange_index = np.random.randint(low=2,size=self.vectors.shape[0])
		exchange_index = exchange_index.astype(bool)

		vectors_a = np.where(exchange_index, self.vectors, mate.vectors)
		vectors_b = np.where(exchange_index, mate.vectors, self.vectors)
		

		return (GeneVector(self.X, self.Y, vectors_a), GeneVector(self.X, self.Y, vectors_b))




class GeneticSearch(object):

	def __init__(self, X, Y, n_genes = 1000, n_interations=10000):
		



		if n_genes % 2 > 0:
			n_genes += 1
			print("n_genes must be an even integer, n_genes is assigned to {} now".format(n_genes))
			
		self.n_genes =  n_genes
		self.n_interations = n_interations



		self.pool = []
		for i in range(n_genes):
			self.pool.append(GeneVector(X,Y))
		self.best_score, self.best_vectors = self.evaluate()

	def getScores(self):
		scores = np.zeros(self.n_genes,float)
		for i in range(self.n_genes):
			scores[i] = self.pool[i].error()
		return scores

	def deterministic_selection(self):
		scores = self.getScores()
		order = np.argsort(scores)

		best_score = scores[order[0]]
		best_vectors = self.pool[order[0]].vectors
		survivors = np.random.shuffle([self.pool[i] for i in order[:50]])
		return survivors

	def tournament_selection(self, tournament_size = 10):
		if tournament_size < 2:
			tournament_size = 2
			print("tournament_size must be bigger than 1, tournament_sizeis assigned to {} now".format(tournament_size))
		
		survivors = []
		scores = self.getScores()
		for i in range(self.n_genes):
			competitors = random.sample(range(self.n_genes), tournament_size)
			competitor_scores = [scores[j] for j in competitors]
			winner_index = np.argmin(competitor_scores)
			winner = self.pool[winner_index]
			survivors.append(winner)
		return survivors
	



	def produce(self, mutation_rate = 1):
		survivors = self.tournament_selection()   # method 1
		for i in range(0, self.n_genes,2):

			reverse_rate = int(1/mutation_rate)
			child_a, child_b = survivors[i].crossover(survivors[i+1])
			if random.randint(0, reverse_rate) == 0:
				child_a.mutate()
			if random.randint(0, reverse_rate) == 0:
				child_b.mutate()


			self.pool[i], self.pool[i+1] = child_a, child_b


	def evaluate(self):
		scores = self.getScores()
		best_score = np.min(scores)
		best_vectors = self.pool[np.argmin(scores)].vectors
		return (best_score, best_vectors)

	def search(self):
		
		for i in range(self.n_interations):
			self.produce()
			score, vectors = self.evaluate()
			if score < self.best_score:
				self.best_score = score
				self.best_vectors = vectors
			print("The {}th generation, error:{}".format(i,self.best_score))







		








df = pd.read_csv("forestfires.txt", index_col=False, sep=" ")
X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values
'''
rv = RandomVectors(X,Y)
print(rv.error())
for i in range(20):
	rv.random_search()
	print(rv.error())
'''
gs = GeneticSearch(X,Y)
gs.search()