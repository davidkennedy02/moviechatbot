from convokit import Corpus, download 
import pandas as pd

def downloadData(filepath:str="movie_dialogues.csv"):
    corpus = Corpus(filename=download("movie-corpus"))

    dialogue_pairs = []
    for conversation in corpus.iter_conversations():
        utterances = list(conversation.iter_utterances())
        for i in range(len(utterances)-1):
            dialogue_pairs.append({
                "input": utterances[i].text,
                "response": utterances[i+1].text
            })
            
    corpus.print_summary_stats()

    df = pd.DataFrame(dialogue_pairs)
    df.to_csv(filepath, index=False)