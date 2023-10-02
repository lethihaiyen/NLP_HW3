from collections import defaultdict, Counter

def open_file(fn) : 
    with open(fn, 'r') as f:
        lines = f.readlines()
    return lines

def save_to_pos_file(data, filename):
    with open(filename, 'w') as file:
        for word, tag in data:
            if word is None:
                file.write('\n')  # Add an empty line for sentence boundary
            else:
                file.write(f"{word}\t{tag}\n")

def calc_proba(lines) : 
    transition_table = defaultdict(Counter)
    prev_tag = None
    indx_tag_counter = Counter() #dict of count initialized default to 0 
    
    for line in lines : 
        line = line.strip() 
        if line: # line not empty
            word,tag = line.split()
            if prev_tag != None : 
                transition_table[prev_tag][tag] +=1
            prev_tag = tag
            indx_tag_counter[tag] +=1 
        # If empty line : tag =  "B"
        else : 
            tag = "B" 
            indx_tag_counter[tag] +=1 
            transition_table[prev_tag][tag] +=1 
            prev_tag = "B"


    # print(indx_tag_counter)
    # Create transition probabilities table
    transition_probabilities = defaultdict(dict)
    for tag, trans_dict in transition_table.items() : 
        for key,value in trans_dict.items() : 
            trans_proba = transition_table[tag][key]/indx_tag_counter[tag]
            transition_probabilities[tag][key] = trans_proba
    

    likelihood_table = defaultdict(Counter)
    for line in lines : 
        line = line.strip() 
        if line : 
            word,tag = line.split()
            likelihood_table[tag][word] += 1

    # Create likelihood probabilities table
    likelihood_proba = defaultdict(dict)
    for tag, freq_dict in likelihood_table.items():
        for word, count in freq_dict.items() : 
            proba_llh = count / indx_tag_counter[tag]
            likelihood_proba[tag][word] =proba_llh

    return transition_probabilities, likelihood_proba

def POS_set(counter) : 
    set_of_POS = set() 
    for pos, count in counter.items() : 
        set_of_POS.add(pos)
    
    return set_of_POS

def train_word_set(lines) : 
    unique_training_word_list = set()

    for line in lines : 
        line = line.strip() 
        if line : 
            word,tag = line.split()
            unique_training_word_list.add(word)
    return unique_training_word_list

def dev_word_set( lines) : 
    word_set = set() 
    for line in lines : 
        line = line.strip() 
        if line : 
            word_set.add(line)

    return word_set

def is_number(s):
    try:
        float(s)
        int(s)
        return True
    except ValueError:
        return False

def OOV(set1, set2, likelihood_proba ) :
    set_ofpos= POS_set(likelihood_proba)
    oov = []
    for word in set1: 
        if word not in set2 : 
            oov.append(word)

    punc = [",", ".", "/","$","``","-","''",".",  "(",")", "[","]","{","}"]
    ends = [ "ish", "ous", "ful", "less", "ble", "ive"]
    for word in oov : 
        # The 
        if word == "The" : 
            likelihood_proba["DT"][word] = 1 
            continue

        # { : }
        if word == "{" : 
            likelihood_proba["{"][word] = 1 
            continue
        # { : }
        if word == "}" : 
            likelihood_proba["}"][word] = 1 
            continue
        # by 
        if word == "by" : 
            likelihood_proba["IN"][word] = 1 
            continue

        # Word is to -> TO
        to = ["to", "To"]
        if word in to : 
            likelihood_proba["TO"][word] = 1
            continue

        # Punctuation
        if word in punc : 
            likelihood_proba[word][word] = 1 
            continue

        # Word as currency  : - NN
        cur = [ "dollar", "Dollar","yen", "Yen", "Euro"] 
        if word in cur : 
            likelihood_proba["NN"][word] = 1
            continue

        # Word as number :  129.25 - CD
        if is_number(word) : 
            likelihood_proba["CD"][word] = 1
            continue

        # Word is & - CC
        if word =="&" : 
            likelihood_proba["CC"][word] = 1
            continue

        # Word capitalize : 
        if word.isupper() : 
            likelihood_proba["NNP"][word] = 1 
            continue    

        # Word start with captial letter : 
        if word[0].isupper() and word[-1] == "s" and len(word) > 2 : 
            likelihood_proba['NNPS'][word] = 0.99
            continue  

        # Word start with captial letter : 
        if word[0].isupper() : 
            likelihood_proba['NNP'][word] = 0.90 
            likelihood_proba['NNN'][word] = 0.10 
            continue  

        # Words ends in "ish", "ous",..."us"
        if len(word) > 2 and word[-2] =="us"  :
            likelihood_proba['JJ'][word] = 1
            continue

        if len(word) >3 and word[-3] in ends : 
            likelihood_proba['JJ'][word] = 1
            continue 

        # Word ends with "s" : 
        if word.endswith("s") : 
            likelihood_proba["NNS"][word] = 2/6
            likelihood_proba["VBZ"][word] = 2/6
            likelihood_proba["NNP"][word] = 1/6
            likelihood_proba["NNPS"][word] = 1/6
            continue               

        # Others
        
        set_ofpos= POS_set(likelihood_proba)
        set_ofpos = { i for i in set_ofpos if len(i)> 1}
        set_ofpos.discard("FW")
        set_ofpos.discard("TO")

        for key in set_ofpos: 
                likelihood_proba[key][word] = 1/10000
    return likelihood_proba

def virtebi(lines,poset, start_prob,likelihood_proba,trans_proba) : 
    txt = [] 
    sentence = [] 
    sentence_POS =[]

    tags = list(poset)

    for line in lines : 
        word = line.strip()
        if word : 
            sentence.append(word)
        else :
            txt.append(sentence)
            sentence = [] 
    
    for sentence in txt :   # for sentence in txt 

        # Create 2D array with dimension (len(row), len(columns) ) 
        virtebi_arr = [[0 for col in range(len(sentence))] for row in range(len(tags))]
        look_up = [[0 for col in range(len(sentence))] for row in range(len(tags))]
                
        # Initialization step
        for tag in tags:
            ind = tags.index(tag)
            virtebi_arr[ind][0] = float(start_prob.get(tag,1e-10)) * likelihood_proba[tag].get(sentence[0], 1e-10)

        # Recursion to find max 
        for n in range(1, len(sentence)):
            for state in range(len(tags)):
                max_prob = 0
                max_state = 0
                for prev_state in range(1,len(tags)):
                    prob = virtebi_arr[prev_state][n-1] * trans_proba[tags[prev_state]].get(tags[state], 1e-10) * likelihood_proba[tags[state]].get(sentence[n], 1e-10)
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_state
                virtebi_arr[state][n] = max_prob
                look_up[state][n] = max_state
        
        # Backtrack to find best path
        path = []
        current_state = max(range(len(tags)), key=lambda st: virtebi_arr[st][-1])
        for t in range(len(sentence) - 1, -1, -1):
            path.append(tags[current_state])
            current_state = look_up[current_state][t]
        path.reverse()
        
        sentence_POS.extend(path)

    return sentence_POS

def merge_result(lines, sentence_POS):
    result = [] 
    pos_index = 0 

    for line in lines : 
        word = line.strip() 
        if word : 
            result.append((word, sentence_POS[pos_index]))
            pos_index += 1
        else:
            result.append((None, None))
    return result

# Calculate the total word in file. - for debugging only
def word_list(dev):
    count = 0
    for l in dev:
        l = l.strip()
        if l:
            count += 1
    return count
# For Debugging : 

def save_diff_to_file(data, filename):
    with open(filename, 'w') as file:
        for line in data : 
            file.write('\n')  # Add an empty line for sentence boundary
            file.write(f"{line}")

# For debugging : diff 
def compare_files(file1, file2):
    wrong_answ = [] 
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        raise ValueError("Files have different number of lines")

    correct = 0
    total = len(lines1)

    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1.strip().split() == line2.strip().split():
            correct += 1
        else:

            wrong_answ.append([f"File1: {line1.strip().split()}\nFile2: {line2.strip().split()}\n"])

    save_diff_to_file(wrong_answ, "wrong_comparision.txt")

    accuracy = correct / total
    print(f"Accuracy: {accuracy*100:.2f}%")
    return accuracy

# Merge File  :
def merge_file( f1, f2, f3) : 
    with open(f1, 'r') as f:
        lines = f.readlines()
    with open(f2, 'r') as f:
        lines2 = f.readlines()
    lines.extend(lines2)
    with open(f3, 'w') as f:
        f.write("".join(lines))   
    return f3


def main():

    # Open File 
    train = "WSJ_02-21.pos"
    dev_file = "WSJ_24.pos"
    train_data = merge_file(train, dev_file, "train_data") 
    testing = "WSJ_23.words"

    # training file & test file
    lines = open_file(train_data)
    test = open_file(testing)

    # Create transmission table, likelihood table from training data 
    trans_proba, likelihood_proba = calc_proba(lines)

    # Define OOV & add to likelihood_proba 
    unique_train_word = train_word_set(lines) 
    unique_development_word = dev_word_set(test) 
    likelihood_proba = OOV(unique_train_word, unique_development_word, likelihood_proba)
    likelihood_proba["}"] = { "}" : 1}
    likelihood_proba["{"] = { "{" : 1}
    likelihood_proba[","] = { "," : 1}
    # Start Probability Initialization. The empty space after "."
    start_prob = trans_proba["B"]

    # Set row for 2D array ( list of POS)
    poset = POS_set(trans_proba)
    # Find POS for each word, merge into POS file and return result.
    pos_arr = virtebi(test,poset, start_prob,likelihood_proba,trans_proba) 
    result = merge_result(test, pos_arr)
    f = save_to_pos_file(result, "submission.pos")   


if __name__ == "__main__":
    main()

