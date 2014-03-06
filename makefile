all:
	R CMD SHLIB normal.cpp
	R CMD SHLIB normal_var.cpp
	R CMD SHLIB normal_em.cpp
	R CMD SHLIB t_gibbs.cpp
