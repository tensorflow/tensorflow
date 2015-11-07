FILE(GLOB Eigen_KroneckerProduct_SRCS "*.h")

INSTALL(FILES
  ${Eigen_KroneckerProduct_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/KroneckerProduct COMPONENT Devel
  )
