subroutine reference_step(x, y) bind(C)
  use iso_c_binding
  use sac_data_mod
  implicit none
  real(c_double), intent(in) :: x(25)
  real(c_double), intent(out) :: y(11)
  real(c_double) :: p(16), s(6), tci, roimp, sdro, ssur, sif, bfs, bfp, tet, deep, gross, factor
  p=x(10:25)
  s=x(4:9)
  RSUM=0
  call SAC1(x(3),x(1),x(2),tci,roimp,sdro,ssur,sif,bfs,bfp,tet,deep, &
       0,0.d0,0.d0,0.d0,0,0.d0, &
       p(1),p(2),p(3),p(12),p(13),p(14),p(9),p(10),p(4),p(6),p(5),p(8),p(7),p(11),p(15),p(16), &
       s(1),s(2),s(3),s(5),s(4),s(6))
  gross=roimp+sdro+ssur+sif+bfs+bfp
  factor=1.d0
  if(gross>0) factor=tci/gross
  y(1:6)=s
  y(7)=(roimp+sdro+ssur)*factor
  y(8)=sif*factor
  y(9)=(bfs+bfp)*factor
  y(10)=tet
  y(11)=deep
end subroutine
