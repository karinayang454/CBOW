# Report

For this assignment, the CBOW model was implemented. 

## Metrics
Visuals of the training and validation performance (in vitro) across epochs can be found in `outputs.png`. All numerical metrics are listed in below:

Train Acc: 21.5%
Val Acc (in vitro): 21.3%
Train Loss: 4.48
Val Loss (in vitro): 4.55

Val (in vivo):
*   Exact (sem): 0.0103
*   MRR (sem): 0.0258
*   MR (sem): 39
*   Exact (syn): 0.0765
*   MRR (syn): 0.1293
*   MR (syn): 8

## Hyperparameters

The model used a `context_size = 4` (left context = 2, right ontext = 2) and a 70/30 train val split. `embedding_dim = 100`, `vocab_size = 3000`, `num_epochs = 20`, `batch_size = 512`, cross entropy loss function, and the `Adam` optimizer with a `lr = 0.01` was applied. Most of these parameters were chosen to match popular online choices when implementing the CBOW model. A high batch size was chosen to decrease training time per epoch.

## Analysis

The in vitro evaluation task uses the validation set to, given n/2 left and n/2 right context words, predict the most likely word (n=4, in this experiment). One simplification is that the accuracy metric only considers the top 1 word, essentially giving equal penalty to both the second and last word choice. Additionally, if the vocab size is set to be very small, one could "cheat" their way to a high accuracy since the <unk> token would be disproportionally high. 

The in vivo task is being evaluated with 3 metrics: Exact, MRR, and MR, for semantic and syntactic relations. "Exact" measures how often the top 1 most similar word was the correct answer (higher the better). Obviously, this metric has similar shortcomings as the accuracy metric in the "in vitro"; other rankings are not considered. "MRR" is the mean reciprocal rank, which is the average of the multiplicative inverse of the rank of the correct answer (higher the better). "MRR" assumes that there is only one relevant answer per query, which in our case, is the correct assumption to make. "MR" is the inverse of "MRR" (lower the better), and can be interpreted as the average rank of where the first correct answer is found. Generally, syntactic relationships are better represented in the word vectors than semantic relationships.

