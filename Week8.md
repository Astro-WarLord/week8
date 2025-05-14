
# MDS613 - Deep Learning - Tutorial Week 8

## Section A: Lecture 8 Review

### 1. What is RNN and how is it different from CNN?
Recurrent Neural Networks (RNNs) are neural networks that are designed to handle sequential data like time-series, text, or speech. They use loops to allow information to persist across steps, enabling memory of previous inputs. In contrast, Convolutional Neural Networks (CNNs) are better suited for spatial data like images.

**Key differences:**  
- RNNs process inputs sequentially while CNNs process spatially.  
- RNNs retain memory of past inputs via hidden states, CNNs do not.  
- RNNs are used in NLP, speech recognition; CNNs in image classification.  

### 2. Types of RNNs with Diagrams and Applications
- **One-to-Many:** Used for text generation.  
- **Many-to-One:** Used for sentiment analysis.  
- **Many-to-Many:** Used for language translation.  

### 3. Long Short-Term Memory (LSTM)
LSTM networks solve the vanishing gradient problem of traditional RNNs by using memory cells and gates.  
- **Forget Gate:** Discards irrelevant info.  
- **Input Gate:** Stores new info.  
- **Output Gate:** Exposes output of cell state.  

### 4. Gated Recurrent Unit (GRU)
GRUs are simplified LSTMs with fewer gates.  
- **Update Gate:** Controls update of activation.  
- **Reset Gate:** Controls how much past info to forget.  

## Section B: Predicting Last Character of a Word

The task is to train an LSTM to predict the last character of a word given the first few characters.

### ✅ Completed Code

```python
new_seq_data = ['sold', 'peep', 'miss', 'told', 'cook', 'hope', 'live', 'mind']
input_batch_new, target_batch_new = make_batch(new_seq_data)
input_batch_new_torch = torch.from_numpy(np.array(input_batch_new)).float().to(device)

net.eval()
with torch.no_grad():
    new_outputs = net(input_batch_new_torch)
    _, new_predicted = torch.max(new_outputs, 1)

print('Predicted characters:', [char_arr[i] for i in new_predicted.cpu().numpy()])
```

### 💬 Comments on Model Performance

The model successfully predicted the last characters for many inputs. However, performance is limited by the small training dataset.

**Suggestions:**  
- Increase dataset size  
- Add dropout layers  
- Use bidirectional LSTM  
- Adjust learning rate and train longer  
