__all__ = ['assemble_matrix_ex0mae',
           'assemble_vector_ex0mae',
           'assemble_residual_ex0mae',
           ' assemble_residual_ex0grad'
]


#==============================================================================
from pyccel.decorators import types

# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_stiffnessmatrix1D(ne, degree, spans, basis, weights, points,  matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v

# assembles mass matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_massmatrix1D(ne, degree, spans, basis, weights, points, matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_0 * bj_0 * wvol

                            matrix[degree+i1, degree+ i2-i1]  += v
    # ...

@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex11(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, matrix):

    # ... sizes
    k1 = weights_1.shape[1]

    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]  
            i_span_2 = spans_2[ie1]      
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):
                            i2 = i_span_2 - p2 + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis_1[ie1, il_1, 1, g1]
                                    bj_0 = basis_2[ie1, il_2, 0, g1]
                                    
                                    wvol = weights_1[ie1, g1]
                                    
                                    v += bi_x * bj_0 * wvol

                            matrix[i1+p1,i2+p2]  += v
    # ...


@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex12(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, matrix):

    # ... sizes
    k1 = weights_1.shape[1]

    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):
                            i2 = i_span_2 - p2 + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis_1[ie1, il_1, 0, g1]
                                    bj_x = basis_2[ie1, il_2, 1, g1]
                                    
                                    wvol = weights_1[ie1, g1]
                                    
                                    v += bi_0 * bj_x * wvol

                            matrix[i1+p1,i2+p2]  += v
    # ...


#==============================================================================
#---Assemble rhs of Picard-Monge-Ampere equation
# with density function equal to sqrt(1.+ |grad(u_h(F))|)
#==============================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_ex0mae(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, vector_u, vector_w, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_u   = zeros((p1+1,p3+1))
    lcoeffs_w   = zeros((p4+1,p2+1))
    # ...
    lvalues_u   = zeros((k1, k2))
    # ...    
    lvalues_u1  = zeros((k1, k2))
    lvalues_u1x = zeros((k1, k2))
    lvalues_u1y = zeros((k1, k2))
    lvalues_u2  = zeros((k1, k2))
    lvalues_u2x = zeros((k1, k2))
    lvalues_u2y = zeros((k1, k2))    
    # ... Assemble a new points by a new map
    int_rhsP    = 0.0
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]

            lvalues_u1[ : , : ]  = 0.0
            lvalues_u1x[ : , : ] = 0.0
            lvalues_u1y[ : , : ] = 0.0
            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p3+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_3[ie2,il_2,0,g2]  #M^p2-1
                            db2  = basis_3[ie2,il_2,1,g2]  #M^p2-1

                            lvalues_u1[g1,g2]  += coeff_u *b1*b2
                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u*b1*db2
            lvalues_u2[ : , : ]  = 0.0
            lvalues_u2x[ : , : ] = 0.0
            lvalues_u2y[ : , : ] = 0.0

            lcoeffs_w[ : , : ]   = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p2+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_4[ie1,il_1,0,g1] #M^p1-1
                        db1  = basis_4[ie1,il_1,1,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2] 
                            db2  = basis_2[ie2,il_2,1,g2] 
                            lvalues_u2[g1,g2]  += coeff_w*b1*b2
                            lvalues_u2x[g1,g2] += coeff_w*db1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2
            lvalues_u[ : , : ] = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    # .. density function
                    rho              = 1.

                    #...
                    u1x              = lvalues_u1x[g1,g2]
                    u1y              = lvalues_u1y[g1,g2]
                    #___
                    u2x              = lvalues_u2x[g1,g2]
                    u2y              = lvalues_u2y[g1,g2]
                    # ...
                    lvalues_u[g1,g2] = sqrt(u1x**2 + u2y**2 + 2. * rho + 2.*u1y**2)
                    # ...
                    wvol             = weights_1[ie1, g1]*weights_2[ie2, g2]
                    int_rhsP        += sqrt(u1x**2 + u2y**2 + 2. * rho + 2.*u1y**2)*wvol 
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p3+1):
                    i1 = i_span_4 - p4 + il_1
                    i2 = i_span_3 - p3 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_4[ie1, il_1, 0, g1] * basis_3[ie2, il_2, 0, g2]
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            #...
                            u     = lvalues_u[g1,g2]
                            #...
                            v    += bi_0 * u * wvol

                    rhs[i1+p4,i2+p3] += v   
    # Integral in Neumann boundary
    int_N = 2.
    # Assuring Compatiblity condition
    coefs = int_N/int_rhsP  
    rhs   = rhs*coefs
    # ...
