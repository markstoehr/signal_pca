import numpy as np
import matplotlib.pyplot as plt

def test_PCA():
    X = np.random.rand(20,4)

def SVD(X):
    """
    Parameters:
    ===========
    X: numpy.ndarray[ndim=2]
        Data matrix, assumption is that rows correspond to observations
        columns correspond to variables, PCA will be done on the
        matrix  1/(n-1) * <(X - mu), (X - mu)>

    Returns:
    ========
    U: numpy.ndarray[ndim=2]
        U is orthonormal such that `X = np.dot(U,np.dot(np.diag(d),V.T))`
    V: numpy.ndarray[ndim=2]
        V is orthonormal such that `X = np.dot(U,np.dot(np.diag(d),V.T))`
    d: numpy.ndarray[ndim=2]
        d is a vector such that the components of d is orthonormal such that `X = np.dot(U,np.dot(np.diag(d),V.T))`

    """
    n,m = X.shape
    Y = (X - np.mean(X,0)) * 1./np.sqrt(n)

    if n >= m:
        w,v = np.linalg.eigh(np.dot(Y.T,Y))
        sort_idx = np.argsort(w)[::-1]
        V,d= v.T[sort_idx].T, np.sqrt(w[sort_idx])
        U = np.dot(Y,V) / d
    else:
        w,v = np.linalg.eigh(np.dot(Y,Y.T))
        sort_idx = np.argsort(w)[::-1][:n]
        U,d= np.dot(Y.T,v.T[sort_idx].T), np.sqrt(w[sort_idx])
        V = np.dot(Y.T,U) / d

    # correct for the fact that we divided by np.sqrt(n)
    d *= np.sqrt(n)

    return U, V, d



def PCA(X,num_pcs=-1):
    """
    Parameters:
    ===========
    X: numpy.ndarray[ndim=2]
        Data matrix, assumption is that rows correspond to observations
        columns correspond to variables, PCA will be done on the
        matrix  1/(n-1) * <(X - mu), (X - mu)>
     
    Returns:
    ========
    pcs: np.ndarray[ndim=2]
        Principal components, these are the vectors that we can use
        to project the data onto a lower-dimensional subspace,
        columns correspond to pc vectors, rows are the variable loadings
        in the vector

        np.dot(X,pcs) is the projection you're looking for

    eigenvalues: np.ndarray[ndim=1]
        eigenvalues indicating the amount of variance the PC accounts for
    """
    n,m = X.shape
    Y = (X - np.mean(X,0)) * 1./np.sqrt(n)

    if n >= m:
        w,v = np.linalg.eigh(np.dot(Y.T,Y))
        sort_idx = np.argsort(w)[::-1]
        pcs,mags= v.T[sort_idx].T, w[sort_idx]
    else:
        w,v = np.linalg.eigh(np.dot(Y,Y.T))
        sort_idx = np.argsort(w)[::-1][:n]
        pcs,mags= np.dot(Y.T,v.T[sort_idx].T), w[sort_idx]

    Hmat = np.dot(pcs,np.diag(mags)**.5)
    cov = np.dot(Y.T,Y)
    Xcov = np.cov(X.T)
    import pdb; pdb.set_trace()

    return pcs,mags


# def plot_pcs(pcs):
#     """
#     Parameters:
#     ===========
#     pcs: np.ndarray[ndim=2]
#        Columns are the principal components
#     """
#     num_pcs = 2
#     pcs = pcs[:,:num_pcs]
#     Z = np.dot(Y,pcs)
#     plt.close('all')
#     plt.scatter(Z[:,0],Z[:,1])

#     transformed_coord_sorted = np.argsort((pcs**2).sum(-1))
#     for i in transformed_coord_sorted[:5]:
#         pc = pcs[i,:]
#         try:
#             plt.arrow(0,0,pc[0],pc[1])
#         except:
#             import pdb; pdb.set_trace()
#         plt.annotate("coordinate %d" %i,xy=(pc[0],pc[1]))
    
#     plt.show()

#     import pdb; pdb.set_trace()
