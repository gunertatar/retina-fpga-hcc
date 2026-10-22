#include "conv_ip.h"
#include "preprocess.h"
#include <cstdio>
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); unsigned char buf[4096]; fread(buf,1,4096,f); fclose(f);
  hls::stream<axis_pixel_t> a,b; hls::stream<axis_fixed_t> o;
  for(int i=0;i<4096;i++){axis_pixel_t p; p.data=buf[i]; p.last=(i==4095); a.write(p);}
  preprocess(a,b); conv_ip(b,o);
  FILE*g=fopen(argv[2],"wb"); for(int i=0;i<32*4096;i++){ short v=(short)o.read().data.to_int(); fwrite(&v,2,1,g);} fclose(g); return 0;}
