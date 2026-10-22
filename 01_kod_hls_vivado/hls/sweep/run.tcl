set T $::env(T); set P $::env(PART); set C $::env(CLK); set U $::env(URAM)
open_project -reset prj_${P}_T${T}
set_top conv_ip
add_files src_${U}_T${T}.cpp
open_solution -reset sol -flow_target vivado
set_part $P
create_clock -period $C
csynth_design
exit
