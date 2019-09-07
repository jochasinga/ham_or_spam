using DataStructures
using Formatting
using ScikitLearn
@sk_import naive_bayes: GaussianNB
@sk_import metrics: accuracy_score

TRAIN_DIR = "./machine-learning-101/chapter1/train-mails"
TEST_DIR = "./machine-learning-101/chapter1/test-mails"

most_common(c::Accumulator) = most_common(c, length(c))
most_common(c::Accumulator, k) = sort(collect(c), by=kv->kv[2], rev=true)[1:k]

function isalpha(str)
    re = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
    !occursin(re, str)
end

"Make a bag of words from email text."
function make_dictionary(root_dir)
    all_words = []
    emails = [joinpath(root_dir, f) for f in readdir(root_dir)]
    for (i, mail) in enumerate(emails)
        println("Opening file: ", i, " ", mail)
        open(mail) do m
            for line in readlines(m)
                words = split(line)
                append!(all_words, words)
            end
        end
    end

    bag = counter(all_words)
    list_to_remove = [k for k in keys(bag)]

    for item in list_to_remove
        # remove if numerical
        if !isalpha(item)
            reset!(bag, item)
            # pop!(bag, item)
        elseif length(item) == 1
            reset!(bag, item)
            # pop!(bag, item)
        end
    end
    # Consider only most 3000 common words
    most_common(bag, 3000)
end

"Make features matrix and label vector"
function extract_features(mail_dir, dict)
    files = [joinpath(mail_dir, f) for f in readdir(mail_dir)]
    features_matrix = zeros(length(files), 3000)
    train_labels = zeros(length(files))
    # cnt = 1
    doc_id = 1
    for (n, file) in enumerate(files)
        open(file) do f
            for (i, line) in enumerate(readlines(f))
                # Skip the first subject line and second empty line
                if i == 3
                    words = split(line)
                    for word in words
                        word_id = 1
                        # Go through the bag of words
                        for (i, d) in enumerate(dict)
                            if d[1] == word
                                printfmt("Found {} in the bag\n", word)
                                word_id = i
                                c = count(x -> x == word, words)
                                features_matrix[doc_id, word_id] = c
                                printfmt("F[{}, {}] = {}\n", doc_id, word_id, c)
                            end
                        end
                    end
                end
            end

            train_labels[doc_id] = 0
            filepath_tokens = split(file, "/")
            last_token = filepath_tokens[length(filepath_tokens)]
            if startswith(last_token, "spmsg")
                train_labels[doc_id] = 1
                # TODO: ?
                # cnt += 1
            end
            doc_id += 1
        end
    end
    (features_matrix, train_labels)
end

dict = make_dictionary(TRAIN_DIR)

(features_matrix, labels) = extract_features(TRAIN_DIR, dict)
(test_features_matrix, test_labels) = extract_features(TEST_DIR, dict)

model = GaussianNB()
# train model
fit!(model, features_matrix, labels)
# predict
predicted_labels = predict(model, test_features_matrix)

accuracy = accuracy_score(test_labels, predicted_labels)

printfmt("accuracy of prediction {}", accuracy)

# println(predicted_labels[132])