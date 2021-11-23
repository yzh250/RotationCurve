

cimport cython

from typedefs cimport DTYPE_F64_t


cpdef DTYPE_F64_t vel_tot_iso(DTYPE_F64_t r, \
                              DTYPE_F64_t log_rhob0, \
                              DTYPE_F64_t Rb, \
                              DTYPE_F64_t SigD, \
                              DTYPE_F64_t Rd, \
                              DTYPE_F64_t rho0_h, \
                              DTYPE_F64_t Rh)

cpdef DTYPE_F64_t vel_tot_NFW(DTYPE_F64_t r, \
                              DTYPE_F64_t log_rhob0, \
                              DTYPE_F64_t Rb, \
                              DTYPE_F64_t SigD, \
                              DTYPE_F64_t Rd, \
                              DTYPE_F64_t rho0_h, \
                              DTYPE_F64_t Rh)

cpdef DTYPE_F64_t vel_tot_bur(DTYPE_F64_t r, \
                              DTYPE_F64_t log_rhob0, \
                              DTYPE_F64_t Rb, \
                              DTYPE_F64_t SigD, \
                              DTYPE_F64_t Rd, \
                              DTYPE_F64_t rho0_h, \
                              DTYPE_F64_t Rh)