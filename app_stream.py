import streamlit as st
st.set_page_config(page_title='MEAT CLASSIFIER')
import PIL
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
class Net(nn.Module):
  def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size = 3),
            nn.ReLU(),
            


            nn.Flatten(),
            nn.Linear(4096,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,3)
            

        )
    
  def forward(self, xb):
        return self.network(xb)


def main():
    st.markdown ("<h1 style='text-align: center; color: orange;'>MEAT STATE CLASSIFIER</h1>", unsafe_allow_html=True)
    st.write('To classify the state of a meat, upload an image and press predict')
    image=st.file_uploader('MEAT IMAGE', type=['png', 'jpg'])
    submitted = st.button("MAKE PREDICTION")
    if submitted:
        image = PIL.Image.open(image)
        st.image(image)
        transform=transforms.Compose( [transforms.ToTensor(), transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ), transforms.Resize((25, 25)) ] )
        image = transform(image).unsqueeze(0)
        model = torch.load('model.pth')
        model = model.cpu()
        model.eval()
        pred = torch.argmax(model(image))
        print(pred)
        pred=pred.item()
        if pred==0:
            st.markdown('## This meat is predicted to be Fresh')
        elif pred==1:
            st.markdown('## This meat is predicted to be HALF-FRESH')
        elif pred== 2:
            st.markdown('## This meat is predicted to be SPOILT')

import streamlit 
import sys
from streamlit import cli as stcli
if __name__ == '__main__':
    if streamlit._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())


