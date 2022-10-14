import argparse, random, os, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from eval_utils import downstream_validation
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import utils
import data_utils
from model import CBOW

random.seed(117)
# python train.py --analogies_fn analogies_v3000_1309.json --data_dir books/ 

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # train val split 70 30
    train_sentences = []
    val_sentences = []
    for sentence in sentences:
        n = random.uniform(0,1)
        train_sentences.append(sentence) if n <=0.7 else val_sentences.append(sentence)

    # create encoded input and output numpy matrices for train and then put them into tensors
    encoded_sentences_train, lens_train = data_utils.encode_data(
        train_sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # create encoded input and output numpy matrices for val and then put them into tensors
    encoded_sentences_val, lens_val = data_utils.encode_data(
        val_sentences,
        vocab_to_index,
        suggested_padding_len,
    )
    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #
    inputs_train, outputs_train = data_utils.create_input_outputs(encoded_sentences_train, lens_train, 2)
    inputs_val, outputs_val = data_utils.create_input_outputs(encoded_sentences_val, lens_val, 2)

    train_data = TensorDataset(torch.from_numpy(inputs_train), torch.from_numpy(outputs_train))
    val_data = TensorDataset(torch.from_numpy(inputs_val), torch.from_numpy(outputs_val))

    train_loader = DataLoader(train_data, shuffle=True, batch_size = args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size = args.batch_size)

    return train_loader, val_loader, index_to_vocab

# 3000,100,4
def setup_model(args, vocab_size, embedding_dim, window_size):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = CBOW(vocab_size, embedding_dim, window_size)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # calculate prediction loss
        # print(pred_logits, torch.flatten(labels))
        loss = criterion(pred_logits, torch.flatten(labels))

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = pred_logits.argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, index_to_vocab = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # build model
    model = setup_model(args, 3000, 100, 4)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"train loss : {train_loss} | train acc: {train_acc}")

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, index_to_vocab)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)


        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)

    fig = plt.figure(figsize=(10,10))

    # training loss
    plt.subplot(2, 2, 1)
    x = list(range(1, len(train_losses) + 1))
    plt.plot(x, train_losses, label = "train_loss")
    plt.gca().set_title("Training Loss per Epoch")
    plt.legend()

    # training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(x, train_accs, label = "train_acc")
    plt.gca().set_title("Training Accuracy per Epoch")
    plt.legend()

    # validation loss
    plt.subplot(2, 2, 3)
    x = list(range(1, len(val_losses) + 1))
    plt.plot(x, val_losses, label = "val_loss")
    plt.gca().set_title("Validation Loss per Epoch")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x, val_accs, label = "val_acc")
    plt.gca().set_title("Validation Accuracy per Epoch")
    plt.legend()

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig("output_figures.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default = 'outputs', help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=20, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    args = parser.parse_args()
    main(args)
