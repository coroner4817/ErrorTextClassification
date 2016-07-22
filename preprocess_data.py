

def preprocess_data(data, min_count):
    unique_ac = list(set(data['ac']))

    ac_list_useful = []
    for iAc in unique_ac:
        if data.ac[data.ac==iAc].count() > min_count:
            ac_list_useful.append(iAc)

    print '[Info]: Selected useful ac count: ', len(ac_list_useful), ' Other manually fill in.'

    tr_tx_data = []
    for (linenum, line) in data.iterrows():
        # in ac_list_useful
        # == 'BLANK SCREEN' or line['ac'] == 'DISTORTED VIDEO'
        if line['ac'] == 'BLANK SCREEN' or line['ac'] == 'DISTORTED VIDEO':
            tr_tx_data.append((line['r'], line['ac']))

    print '[Status]: Finish pre-processing'

    return tr_tx_data
