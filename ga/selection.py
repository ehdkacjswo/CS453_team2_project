# Selct next generation saving some of current population
def save_sel(output, new_output, leaf_ind, p, save_p):
	output[leaf_ind].extend(new_output)
	output[leaf_ind] = output[leaf_ind][:save_p] + sorted(output[leaf_ind][save_p:], key=lambda data: data[1][leaf_ind])[:p - save_p]
	output[leaf_ind] = sorted(output[leaf_ind], key=lambda data: data[1][leaf_ind])
