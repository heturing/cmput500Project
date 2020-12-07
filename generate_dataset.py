# This file provides functions that helps to generate a dataset from the groud truth.
from argparse import ArgumentParser
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np


def process_args():
    # All arguments are processed in this function.
    parsar = ArgumentParser()
    parsar.add_argument("-df", "--data_file", dest="data_file", help="The path to the ground truth file", required=True)
    parsar.add_argument("-o", "--output", dest="output", help="Specify the output file", required=False, default="output.txt")
    parsar.add_argument("-mf", "--model_file", dest="model_file", help="Where to save the word embedding model", required=False, default="word_embedding.model")
    parsar.add_argument("-v", "--verbose", dest="verbose", help="How detail you want the output be", required=False, default="1")
    parsar.add_argument("-we", "--word_embedding", dest="word_embedding", help="Which word embedding model to use", required=False, default="Word2Vec")
    args = parsar.parse_args()
    return args

def get_datafacts(s):
    temp = s.split(":")[1].split(" ")[1:-1]
    return temp

def get_token(e):
    # Given a jimple expr, tokenize the expr.
    res = []
    temp = e.split(" ")

    for t in temp:
        t0 = t.replace(">", "<").split("<")
        res.extend([tok.strip(".") for tok in t0 if tok != ""])
    return res

def is_num_token(t):
    if t.isnumeric():
        return True
    if t[0] == "(" and t[-1] == ")":
        return t[1:-1].isnumeric()
    else:
        return False
        
def create_word_embedding_model(word_embedding, corpus):
    if word_embedding == "Word2Vec":
        model = Word2Vec(corpus, min_count = 1)
    elif word_embedding == "Doc2Vec":
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
        model = Doc2Vec(documents, min_count = 1)
    else:
        raise RuntimeError("no word embedding model")
    return model



def main(data_file, output, model_file, verbose, word_embedding):

    # Initialize data structure
    all_datafacts = set()
    all_tokens = set()
    corpus = []
    training_example = []

    # A list that store all tuples of (tokens, in_data, out_data) as intermediate result
    intermediate_training_set = []

    # A map that stores all class_method and its datafacts mapping
    all_datafacts_mapping = {}
    cnt = 0

    with open(data_file) as f:
        # Read each line of ground truth
        all_funcs = []
        func_lines= []
        for line in f:
            # Group exprs as functions they belong

            # Exclude non-data line
            if line == "":
                continue
            if line[:5] == "Class":
                class_method_info = line
                continue
            

            # If we neet the next function and there are some exprs about the previous function.
            elif line[0] == "*":
                if not func_lines:
                    continue     

                datafacts_in_func = set()
                tokens_in_out = []

                # Process the whole previous function
                for func_line in func_lines:

                    # 1. Split the line into expr, in_data, and out_data
                    # Split the string.
                    temp = func_line.split("\t")

                    # Get jimple code
                    expr = temp[0].rstrip(".")
                    
                    # 2. Gather all datafacts in this function
                    input_datafacts = get_datafacts(temp[1])
                    output_datafacts = get_datafacts(temp[2])

                    datafacts_in_func = datafacts_in_func.union(set(input_datafacts))
                    datafacts_in_func = datafacts_in_func.union(set(output_datafacts))

                    # 2.5 Process irrelevant data. string => "STR"
                    if expr.find('"') != -1:
                        fst = expr.find('"')
                        snd = expr.find('"', fst+1)
                        expr = expr.replace(expr[fst-1:snd+2], "STR")


                    # 3. Tokenize the expr, Get rid of < and >
                    tokens = get_token(expr)


                    # 4. Process irrelevant data. 
 
                    # Traverse the tokens and replace the type
                    for i in range(len(tokens)):
                        # Type names (Class type, primitive type) => "TYPE"
                        # An observation, if a token is end with ':'. then this token and the next token is a type. 
                        # One exception is when this token starts with '@', then only the next token is a type.
                        
                        if tokens[i][-1] == ':':
                            tokens[i+1] = "TYPE"
                            if tokens[i][0] != '@':
                                tokens[i] = "TYPE"
                        # Also, the token follows 'new' or 'init' is a type
                        if tokens[i] == "new" or tokens[i] == "init":
                            tokens[i+1] = "TYPE"


                        # numbers (int, float, double) => "NUM"
                        if is_num_token(tokens[i]):
                            tokens[i] = "NUM"

                        # {virtualinvoke, staticinvoke, specialinvoke} -> "INVOKE"
                        if tokens[i] in ["virtualinvoke", "staticinvoke", "specialinvoke"]:
                            tokens[i] = "invoke"

                        # @parameter0,1,2,... => "PARAM"
                        if tokens[i][0] == '@':
                            tokens[i] = "PARAM"

                        # Method Signature => "MTHDSIG"
                        if tokens[i][0] != "(" and tokens[i].find("(") != -1:
                            tokens[i] = "MTHDSIG"

                    tokens_in_out.append((tokens, input_datafacts, output_datafacts))
           
                
                # 6. Build a map that renames all the data facts. New data facts should have the form "DF_"+num
                if verbose == "1":
                    pass
                    #print("There are " + str(len(datafacts_in_func)) + " different data facts in this function.")
                new_data_facts = ["DF_" + str(i) for i in range(len(datafacts_in_func))]
                map_to_new_data_facts = dict(zip(list(datafacts_in_func), new_data_facts))
                all_datafacts_mapping[class_method_info] = map_to_new_data_facts
                if verbose == "1":
                    pass
                    #print("The map from old data facts to new data facts is:")
                    #print(map_to_new_data_facts)
                

                # 7. Save intermediate result: (tokens, in_fact, out_fact)
                for tok in tokens_in_out:
                    in_facts = [map_to_new_data_facts[i] for i in tok[1]]
                    out_facts = [map_to_new_data_facts[i] for i in tok[2]]
                    final_token = tok[0]
                    for j in range(len(final_token)):
                        if final_token[j] in map_to_new_data_facts.keys():
                            final_token[j] = map_to_new_data_facts[final_token[j]]

                    intermediate_training_set.append((final_token, in_facts, out_facts))
                        

                # After processing, remove all exprs for previous function
                func_lines = []
            else:
                func_lines.append(line)
                cnt += 1

        
    # All (tokens, in_fact, out_fact) data should be store into a list
    if verbose == "1":
        print("Intermediate training set:")
        #for i in intermediate_training_set:
        #    print(i)
        print("There are " + str(len(intermediate_training_set)) + " records in intermediate set:")
        print(str(len(all_datafacts_mapping)) + " mappings are stored.")

    # Extract all tokens into a list and train the word embedding model
    for training_example in intermediate_training_set:
        corpus.append(training_example[0])
    word_embedding_model = create_word_embedding_model(word_embedding, corpus)
    


    # Save the model
    word_embedding_model.save(model_file)

    # Gather all new data facts
    all_new_datafacts = set()
    for training_example in intermediate_training_set:
        all_new_datafacts = all_new_datafacts.union(set(training_example[1]))
        all_new_datafacts = all_new_datafacts.union(set(training_example[2]))



    # map in_data and out_data into a vectot of {0,1}^n, Generate the training data (in_vec, out_vec)
    
    
    output_dataset = []
    for training_example in intermediate_training_set:
        times = 0
        vec = np.zeros((100,))

        in_facts_vec = [0] * len(all_new_datafacts) 
        for df in training_example[1]:
            in_facts_vec[list(all_new_datafacts).index(df)] = 1

        out_facts_vec = [0] * len(all_new_datafacts)
        for df in training_example[2]:
            out_facts_vec[list(all_new_datafacts).index(df)] = 10

        # Encode with Word2Vec. Get vector for each token and average them

        if word_embedding == "Word2Vec":
            for tok in training_example[0]:
                vec += word_embedding_model.wv[tok]
                times += 1

            in_vec = list(vec / times)

        # Encode with Doc2Vec. Get the vector for all tokens directly. 
        if word_embedding == "Doc2Vec":
            in_vec = list(word_embedding_model.infer_vector(training_example[0]))

        in_vec.extend(in_facts_vec)
        in_vec = np.array(in_vec)
        out_vec = np.array(out_facts_vec)

        output_dataset.append((in_vec, out_vec))

    if verbose == "1":
        print("Output dataset contains " + str(len(output_dataset)) + " training example.")
        print("Input vector has length " + str(output_dataset[0][0].shape[0]))


    # Before run the dataset, should 1. merge tokens and in_fact_vec into in_vec. 2. cast in_vec and out_vec into ndarray 3. Split x and y

    return output_dataset


