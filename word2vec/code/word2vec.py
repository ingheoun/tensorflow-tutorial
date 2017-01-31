# import tensorflow
import tensorflow as tf
#import module
import base
import collections
import random
import numpy as np

url = 'http://mattmahoney.net/dc/'
filename = 'text8.zip'
expected_bytes = 31344016
t_sub = 10**-5				# Subsampling threshold 
context_size = 2
BOW_size = 5000
hidden_node_size = 128
batch_size = 128
k_neg = 128									# Negative sample size
lr = 0.01
epoch = 400000
plot_only = 500				# < BOW_size

# Tensorflow interactive session 
sess = tf.InteractiveSession()

# Step 1: Download the data.
filepath = base.maybe_download(filename, expected_bytes, url)

# Read the data into a list of strings.
words = base.read_data(filename)
print('Data size', len(words))

word_cnt = collections.Counter(words).most_common(BOW_size)

# numbering a BOW words
word_dict = {}					#key:num
rev_word_dict = {}			#key:word

for i, (word, _) in enumerate(word_cnt):
	word_dict[i] = word
	rev_word_dict[word] = i

# P_n: NEG distribution / Pw_dict: Subsampling (removal) frequency
P_n, Pw_dict = base.calculate_NEG_sub_distribution(word_cnt, BOW_size, t_sub)

# Subsample the frequent words and clean up the words-data 
temp = []
for word in words:
	if word in rev_word_dict:
		if random.random() > Pw_dict[word]:				# with a prob of (1- Pw_dict[word])
			temp.append(word)
words = temp
del temp
print('Sorted data size', len(words))

# make a training set
train_set = []
for i in range(context_size, len(words)-context_size):
	for j in range(-context_size,context_size+1):
		if j != 0:
			if (words[i] in rev_word_dict) and (words[i+j] in rev_word_dict):
				train_set.append((rev_word_dict[words[i]], rev_word_dict[words[i+j]]))

random.shuffle(train_set)
print('Train size', len(train_set))

del words

# make inputs, labels and negative samples
word_in = tf.placeholder(tf.int32, shape=[batch_size])
word_out = tf.placeholder(tf.int32, shape=[batch_size])
word_neg = tf.placeholder(tf.int32, shape=[batch_size, k_neg]) 

# make weights
W_in = tf.Variable(tf.random_uniform([BOW_size, hidden_node_size], -1.0, 1.0))		#(BOW_size, hidden_node_size,)
W_out = tf.Variable(tf.random_uniform([BOW_size, hidden_node_size], -1.0, 1.0))		#(BOW_size, hidden_node_size,)

loss = tf.constant(0.0)
average_loss = 0
# Initialize the variables
sess.run(tf.global_variables_initializer())

# Calculate the Negative Sampling objective with log of sigmoid		
v_in = tf.nn.embedding_lookup(W_in, word_in)															#(batch_size, hidden_node_size,)
v_out = tf.nn.embedding_lookup(W_out, word_out)														#(batch_size, hidden_node_size,)	
v_neg = tf.nn.embedding_lookup(W_in, tf.transpose(word_neg))							#(k_neg, batch_size, hidden_node_size,), transpose for broadcasting

data = tf.reduce_sum(tf.mul(v_in, v_out), axis = 1)												#(batch_size,)
noise_tr = tf.reduce_sum(tf.mul(v_in, v_neg), axis = 2)										#(k_neg, batch_size,) 
noise = tf.transpose(noise_tr)																						#(batch_size, k_neg,)

loss_data = tf.log(tf.sigmoid(data))
loss_noise = tf.log(tf.sigmoid(-noise))

loss = tf.reduce_sum(loss_data) + tf.reduce_sum(loss_noise)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(-loss)
saver = tf.train.Saver()
print ("Train start!")

# assign batch by feed_dict 
for i in range(epoch):
	train_input, labels = base.next_batch_in_index(batch_size=batch_size, train_set = train_set, BOW_size=BOW_size)
	train_neg = np.array([base.choose_NEG_sample(P_n) for _ in range(batch_size * k_neg)]).reshape(batch_size, k_neg)
	_, loss_val = sess.run([train_step, loss], feed_dict={word_in: train_input, word_out: labels, word_neg: train_neg})
	average_loss += loss_val
	if i%200 == 0:
		print("Epoch: ", i)
		average_loss /= 200
		print("Average loss: ", average_loss)
		average_loss = 0
	if i%2000 == 0:
		#save the model
		saver.save(sess, 'Lookup/my-model', global_step=i)
		
		# Plot the pca graph 
		norm = tf.sqrt(tf.reduce_sum(tf.square(W_in), 1, keep_dims=True))
		normalized_embeddings = W_in / norm
		final_embeddings = normalized_embeddings.eval()
		try:
			from sklearn.manifold import TSNE
			import matplotlib.pyplot as plt
			
			tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
			low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
			labels = [word_dict[i] for i in range(plot_only)]
			filename = 'Lookup/word2vec_Lookup_' + str(i) + '.png'
			base.plot_with_labels(low_dim_embs, labels, filename=filename)
			print("saved!")
		except ImportError:
			print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
	

sess.close()



