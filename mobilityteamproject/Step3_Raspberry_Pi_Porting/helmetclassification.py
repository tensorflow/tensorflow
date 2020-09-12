from PIL import Image



from tflite_runtime.interpreter import Interpreter



from tflite_runtime.interpreter import load_delegate



from video import create_capture







import numpy as np



import cv2 as cv



import io



import picamera



import simpleaudio as sa











# tf model upload



def load_labels(path):



  with open(path, 'r') as f:



    return {i: line.strip() for i, line in enumerate(f.readlines())}















def set_input_tensor(interpreter, image):



  tensor_index = interpreter.get_input_details()[0]['index']



  input_tensor = interpreter.tensor(tensor_index)()[0]



  input_tensor[:, :] = image











# check whether user wears helmet 



def classify_image(interpreter, image, top_k=1):



  set_input_tensor(interpreter, image)



  interpreter.invoke()



  output_details = interpreter.get_output_details()[0]



  output = np.squeeze(interpreter.get_tensor(output_details['index']))















  # If the model is quantized (uint8 data), then dequantize the results



  if output_details['dtype'] == np.uint8:



    scale, zero_point = output_details['quantization']



    output = scale * (output - zero_point)



  ordered = np.argpartition(-output, top_k)







  # if 0.90 above then regard user is wearing a helmet



  if (top_k==1) and (output[1] > 0.9):



    res = 1



  else:



    res = 0



  return res











# for detect human face



def detect(img, cascade):



    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)







    if len(rects) == 0:



        return []



    rects[:,2:] += rects[:,:2]



    return rects











def main():



    import sys, getopt

    checknum = 0



    while True:



        try:



          # face recognizing code



          print('face camera ')



          args, video_src = getopt.getopt(sys.argv[1:2], '', ['cascade=', 'nested-cascade='])



          try:



              video_src = video_src[0]



          except:



              video_src = 0







          args = dict(args)



          cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")



          nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")



          



          cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))



          nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))



          cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/lena.jpg')))



                



          while True:



              ret, img = cam.read()



              gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



              gray = cv.equalizeHist(gray)







              rects = detect(gray, cascade)



              vis = img.copy()



              if len(rects):



                  if not nested.empty():



                      print('into nested') # 사람이 들어왔을 때



                      for x1, y1, x2, y2 in rects:



                          roi = gray[y1:y2, x1:x2]



                          vis_roi = vis[y1:y2, x1:x2]



                          print('findrects')



                          subrects = detect(roi.copy(), nested)







                      if subrects!='[]':



                          faceok = 'faceok.wav'



                          fa = sa.WaveObject.from_wave_file(faceok)



                          face = fa.play()



                          face.wait_done()



                          print('detect!!')



                          break



            



          cam.release() # face recognition camera off



          print("helmet camera")



					



					# helmet detectecting code



          filename = 'helmet.wav'



          wave_obj = sa.WaveObject.from_wave_file(filename)



          helmetok = 'helmetok.wav'



          wave = sa.WaveObject.from_wave_file(helmetok)







          labels = "labels.txt"



          model = "model_edgetpu.tflite"



          interpreter = Interpreter(model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])



          interpreter.allocate_tensors()



          _, height, width, _ = interpreter.get_input_details()[0]['shape']







          # helmet detect camera on



          with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:



              camera.start_preview()



              try:



                  stream = io.BytesIO()



                  for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):



                      stream.seek(0)



                      image = Image.open(stream).convert('RGB').resize((width, height),Image.ANTIALIAS)



                      results = classify_image(interpreter, image)



                      print("result:")



                      print(results)



                      



                      stream.seek(0)



                      stream.truncate()







                      # 헬멧 착용여부 판단



                      if results==0:



                          play_obj = wave_obj.play()



                          play_obj.wait_done()

                          checknum += 1

                          if checknum==3:

                              checknum = 0

                              break;



                  



                      else:



                          helm = wave.play()



                          helm.wait_done()



                          print('GoodBoy')



                          break



                  



              finally:



                  camera.stop_preview()







        except KeyboardInterrupt:



            break











if __name__ == '__main__':



    main()



    cv.destroyAllWindows()
