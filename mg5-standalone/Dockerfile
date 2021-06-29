FROM rootproject/root:6.22.00-ubuntu20.04

USER root

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
wget ca-certificates python python-dev gfortran build-essential \
ghostscript nano vim libboost-all-dev rsync gnuplot
RUN wget --quiet -O- https://bootstrap.pypa.io/get-pip.py | python3 -
RUN pip3 install numpy scipy pandas matplotlib seaborn atlas-mpl-style

WORKDIR /home/hep

ENV MG_VERSION="MG5_aMC_v2_9_1_2"

RUN wget --quiet -O- https://launchpad.net/mg5amcnlo/2.0/2.9.x/+download/MG5_aMC_v2.9.1.2.tar.gz | tar xzf -
WORKDIR /home/hep/${MG_VERSION}

ENV ROOTSYS /usr/local
ENV PATH $PATH:$ROOTSYS/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROOTSYS/lib

# install tools
RUN echo "install lhapdf6" | /home/hep/${MG_VERSION}/bin/mg5_aMC
RUN echo "install mg5amc_py8_interface" | /home/hep/${MG_VERSION}/bin/mg5_aMC

# disable autoupdate
RUN rm /home/hep/${MG_VERSION}/input/.autoupdate

# install PDF
WORKDIR /home/hep/${MG_VERSION}/HEPTools/lhapdf6_py3/share/LHAPDF
# Download few default PDFs
RUN wget --quiet -O- http://lhapdfsets.web.cern.ch/lhapdfsets/current/NNPDF23_lo_as_0130_qed.tar.gz | tar xzf -
RUN wget --quiet -O- http://lhapdfsets.web.cern.ch/lhapdfsets/current/NNPDF30_lo_as_0130.tar.gz | tar xzf -
RUN wget --quiet -O- http://lhapdfsets.web.cern.ch/lhapdfsets/current/NNPDF31_lo_as_0130.tar.gz | tar xzf -

# install Delphes
# (must install pre-release due to missing TF1 include in current version 4.2)
WORKDIR /home/hep/${MG_VERSION}/HEPTools
RUN wget --quiet -O- https://github.com/delphes/delphes/archive/3.4.3pre08.tar.gz | tar xzf - && cd delphes-3.4.3pre08 && make
ENV LD_LIBRARY_PATH /home/hep/${MG_VERSION}/HEPTools/delphes-3.4.3pre08/:$LD_LIBRARY_PATH
ENV ROOT_INCLUDE_PATH /home/hep/${MG_VERSION}/HEPTools/delphes-3.4.3pre08/external/:$ROOT_INCLUDE_PATH

# install Fastjet
RUN wget --quiet -O- http://fastjet.fr/repo/fastjet-3.3.4.tar.gz | tar xzf - && cd fastjet-3.3.4/ && ./configure --prefix=$PWD/build && make && make check && make install
ENV LD_LIBRARY_PATH /home/hep/${MG_VERSION}/HEPTools/fastjet-3.3.4/build/lib/:$LD_LIBRARY_PATH

# set up config executables
RUN echo "set lhapdf /home/hep/${MG_VERSION}/HEPTools/lhapdf6_py3/bin/lhapdf-config" | /home/hep/${MG_VERSION}/bin/mg5_aMC
RUN echo "set delphes_path /home/hep/${MG_VERSION}/HEPTools/delphes-3.4.3pre08/" | /home/hep/${MG_VERSION}/bin/mg5_aMC

# set up MadGraph
RUN echo "set auto_convert_model True" | /home/hep/${MG_VERSION}/bin/mg5_aMC
RUN ln /home/hep/MG5_aMC_v2_9_1_2/bin/mg5_aMC /usr/bin/mg5_aMC

# install models
COPY . /home/hep/
RUN chmod 777 /home/hep/${MG_VERSION}/models
RUN cp -r /home/hep/Models/* /home/hep/${MG_VERSION}/models/

# set workdir and environment variables
WORKDIR /var/MG_outputs
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/home/hep/${MG_VERSION}/HEPTools/lib/
CMD /bin/bash
