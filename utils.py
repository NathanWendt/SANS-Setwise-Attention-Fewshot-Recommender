def get_loss_weight(u, support_set, data, alpha, c0):

    Iu_sup = len(support_set)
    Iu_all = len(data[u]['pos'])
    cu = c0 * (Iu_sup ** alpha) / (Iu_all ** alpha)

    return cu