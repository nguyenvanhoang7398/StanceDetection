import os


def build_mrpc(mrpc_root):
    print("Processing MRPC...")
    mrpc_train_file = os.path.join(mrpc_root, "msr_paraphrase_train.txt")
    mrpc_test_file = os.path.join(mrpc_root, "msr_paraphrase_test.txt")
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file

    dev_ids = []
    with open(os.path.join(mrpc_root, "mrpc_dev_ids.tsv"), encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file, encoding="utf8") as data_fh, \
            open(os.path.join(mrpc_root, "mrpc.train"), 'w', encoding="utf8") as train_fh, \
            open(os.path.join(mrpc_root, "mrpc.val"), 'w', encoding="utf8") as dev_fh:
        header = data_fh.readline()
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s-%s\t%s\t%s\t%s\n" % (id1, id2, s1, s2, label))
            else:
                train_fh.write("%s-%s\t%s\t%s\t%s\n" % (id1, id2, s1, s2, label))

    with open(mrpc_test_file, encoding="utf8") as data_fh, \
            open(os.path.join(mrpc_root, "mrpc.test"), 'w', encoding="utf8") as test_fh:
        header = data_fh.readline()
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%s-%s\t%s\t%s\t%s\n" % (id1, id2, s1, s2, label))
    print("\tCompleted!")
