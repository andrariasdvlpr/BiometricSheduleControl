from django.shortcuts import render

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from bsc.permissions import IsOptionsOrIsAuthenticated

from django.contrib.auth.models import User
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from biometric.utils import aling_image, encrypt_image,dencrypt_image_facenet,dencrypt_image
from core.models import ActivityLogType, Employees, ActivityLog, ActivityLogState

from datetime import datetime
from datetime import date

import base64
import json

import os, errno
import facenet
from .apps import BiometricConfig

class DoFacialVerification(APIView):

    permission_classes = (IsOptionsOrIsAuthenticated,)

    def post(self, request,type_activity_log):
        try:
            #save temp image 
            dir_path = os.path.join(settings.BASE_DIR, "temp")
            image_file = request.FILES['imagen']
            fs = FileSystemStorage(location=dir_path)
            uploaded_file_url = fs.save(image_file.name, image_file)
            path_temp = os.path.join(dir_path,uploaded_file_url)

            #myfileVoice = request.FILES['voiceRecord']
            #fsVoice = FileSystemStorage(location=dir_path)
            #uploaded_voice = fsVoice.save(myfileVoice.name, myfileVoice)

            location_json = request.data['location']

            #align image and get user template url
            core_users = User.objects.get(username= request.user.username)
            path_img = aling_image(path_temp_image=path_temp,image_name=uploaded_file_url,username=str(core_users.employees.id),save_directory="log\\")
            
            #delete temporal image
            if os.path.exists(path_temp):
                os.remove(path_temp)

            aux_path = core_users.employees.user_template.url
            path_template = aux_path.replace("%3A",":")
            dencrypt_image_facenet(path_template,core_users.employees.employee_id, path_temp)

            
            # Get input and output tensors
            sess = BiometricConfig.sess
            images_placeholder = BiometricConfig.images_placeholder
            embeddings = BiometricConfig.embeddings
            phase_train_placeholder = BiometricConfig.phase_train_placeholder
    
            paths_batch =[path_temp,path_img]
            images = facenet.load_data(paths_batch, False, False, 160)

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    
            dist = facenet.distance(emb[0::2], emb[1::2],distance_metric=1)
            now = datetime.now()

            states = ActivityLogState.objects.all().order_by('-sc_threshold')

            states[len(states)-1].sc_threshold = core_users.employees.decision_threshold
            
            sc_state = 0
            for state in states:
                if dist < state.sc_threshold:
                    sc_state = state.sc_state_id

            sc_type = ActivityLogType.objects.get(sc_type_id=type_activity_log)
            sc = ActivityLog(employee=core_users.employees,type=sc_type,sc_date=now,sc_distance=dist[0],sc_state=sc_state)
            sc.sc_photo = path_img
            sc.sc_location = location_json
            sc.save()

            encrypt_image(path_img,core_users.employees.employee_id)

            content ={'first_name': core_users.first_name, 'last_name': core_users.last_name, 'employee_id':core_users.employees.employee_id,
            'datetime':now, 'distance':dist[0], "state": sc_state}
            state = status.HTTP_200_OK
        except Exception as e :
            content = {"message" : "Algo inesperado ha ocurrido", "Exception" : str(e)}
            state = status.HTTP_500_INTERNAL_SERVER_ERROR
        finally:
            if os.path.exists(path_temp):
                os.remove(path_temp)

            return Response(content, status= state)

class SetUserTemplate(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def post(self, request):
        try:
            #save temp image 
            dir_path = os.path.join(settings.BASE_DIR, "temp")
            image_file = request.FILES['imagen']
            fs = FileSystemStorage(location=dir_path)
            uploaded_file_url = fs.save(image_file.name, image_file)
            path_temp = os.path.join(dir_path,uploaded_file_url)

            user = User.objects.get(username= request.user.username)
            path_img = aling_image(path_temp_image=path_temp,image_name=uploaded_file_url,username=str(user.employees.id),save_directory="tmplt\\")
            
            core_users = user.employees
            core_users.user_template=path_img
            core_users.save()
            encrypt_image(path_img,core_users.employee_id)

            if os.path.exists(path_temp):
                os.remove(path_temp)

            template = core_users.user_template
            img_B64_Template = "null"  
            if template :
                aux_path = template.url
                img_B64_Template = dencrypt_image(aux_path.replace("%3A", ":"),core_users.employee_id)
            
            content ={'employee_id':core_users.employee_id,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'username' : user.username,
            'email' : user.email,
            'image': img_B64_Template,
            'registration_date' : core_users.registration_data}
            state = status.HTTP_200_OK
        except Exception as e :
            content = {"message" : "Algo inesperado ha ocurrido", "Exception" : str(e)}
            state = status.HTTP_500_INTERNAL_SERVER_ERROR
        finally:
            if os.path.exists(path_temp):
                os.remove(path_temp)

            return Response(content, status= state)


class DoAsynchronFacialVerification(APIView):
    
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def post(self, request,type_activity_log,date_time):
        try:
            #save temp image 
            dir_path = os.path.join(settings.BASE_DIR, "temp")
            image_file = request.FILES['imagen']
            fs = FileSystemStorage(location=dir_path)
            uploaded_file_url = fs.save(image_file.name, image_file)
            path_temp = os.path.join(dir_path,uploaded_file_url)

            #myfileVoice = request.FILES['voiceRecord']
            #fsVoice = FileSystemStorage(location=dir_path)
            #uploaded_voice = fsVoice.save(myfileVoice.name, myfileVoice)

            location_json = request.data['location']

            #align image and get user template url
            core_users = User.objects.get(username= request.user.username)
            path_img = aling_image(path_temp_image=path_temp,image_name=uploaded_file_url,username=str(core_users.employees.id),save_directory="log\\")
            
            #delete temporal image
            if os.path.exists(path_temp):
                os.remove(path_temp)
            
            aux_path = core_users.employees.user_template.url
            path_template = aux_path.replace("%3A",":")
            dencrypt_image_facenet(path_template,core_users.employees.employee_id, path_temp)
            
            # Get input and output tensors
            sess = BiometricConfig.sess
            images_placeholder = BiometricConfig.images_placeholder
            embeddings = BiometricConfig.embeddings
            phase_train_placeholder = BiometricConfig.phase_train_placeholder
    
            paths_batch =[path_temp,path_img]
            images = facenet.load_data(paths_batch, False, False, 160)

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    
            dist = facenet.distance(emb[0::2], emb[1::2],distance_metric=1)
            now = datetime.fromtimestamp(date_time/1000.0)

            states = ActivityLogState.objects.all().order_by('-sc_threshold')

            states[len(states)-1].sc_threshold = core_users.employees.decision_threshold
            
            sc_state = 0
            for state in states:
                if dist < state.sc_threshold:
                    sc_state = state.sc_state_id

            sc_type = ActivityLogType.objects.get(sc_type_id=type_activity_log)
            sc = ActivityLog(employee=core_users.employees,type=sc_type,sc_date=now,sc_distance=dist[0],sc_state=sc_state)
            sc.sc_photo = path_img
            sc.sc_location = location_json
            sc.save() 

            encrypt_image(path_img,core_users.employees.employee_id)

            content ={'first_name': core_users.first_name, 'last_name': core_users.last_name, 'employee_id':core_users.employees.employee_id,
            'datetime':now, 'distance':dist[0], "state": sc_state}
            state = status.HTTP_200_OK
        except Exception as e :
            content = {"message" : "Algo inesperado ha ocurrido", "Exception" : str(e)}
            state = status.HTTP_500_INTERNAL_SERVER_ERROR
        finally:
            if os.path.exists(path_temp):
                os.remove(path_temp)

            return Response(content, status= state)

class DoFacialVerificationWebApp(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def post(self, request,type_activity_log):
        #save temp image 
        dir_path = os.path.join(settings.BASE_DIR, "temp")
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                #directory already exists
                pass

        received_json_data=json.loads(request.body)
        image_file = received_json_data['imagen']
        uploaded_file_url = str("IMG_"+str(datetime.now().timestamp()).replace(".","")+"_"+str(datetime.now().year)+str(datetime.now().month)+".jpg")
        path_temp = os.path.join(dir_path,uploaded_file_url)
        img_B64 = image_file.replace("data:image/png;base64,","")
        img_wr = base64.b64decode(img_B64)
        with open(path_temp, 'wb') as image_Saved:
            image_Saved.write(img_wr)

        location_json = json.dumps(received_json_data['location'])
        #align image and get user template url
        core_users = User.objects.get(username= request.user.username)
        path_img = aling_image(path_temp_image=path_temp,image_name=uploaded_file_url,username=str(core_users.employees.id),save_directory="log\\")
        
        #delete temporal image
        if os.path.exists(path_temp):
            os.remove(path_temp)
        
        aux_path = core_users.employees.user_template.url
        path_template = aux_path.replace("%3A",":")
        dencrypt_image_facenet(path_template,core_users.employees.employee_id, path_temp)
        
        # Get input and output tensors
        sess = BiometricConfig.sess
        images_placeholder = BiometricConfig.images_placeholder
        embeddings = BiometricConfig.embeddings
        phase_train_placeholder = BiometricConfig.phase_train_placeholder
  
        paths_batch =[path_temp,path_img]
        images = facenet.load_data(paths_batch, False, False, 160)

        # Run forward pass to calculate embeddings
        feed_dict = { images_placeholder: images, phase_train_placeholder:False }
        emb = sess.run(embeddings, feed_dict=feed_dict)
 
        dist = facenet.distance(emb[0::2], emb[1::2],distance_metric=1)
        now = datetime.now()

        states = ActivityLogState.objects.all().order_by('-sc_threshold')

        states[len(states)-1].sc_threshold = core_users.employees.decision_threshold
        
        sc_state = 0
        for state in states:
            if dist < state.sc_threshold:
                sc_state = state.sc_state_id

        sc_type = ActivityLogType.objects.get(sc_type_id=type_activity_log)
        sc = ActivityLog(employee=core_users.employees,type=sc_type,sc_date=now,sc_distance=dist[0],sc_state=sc_state)
        sc.sc_photo = path_img
        sc.sc_location = location_json
        sc.save()

        encrypt_image(path_img,core_users.employees.employee_id)

        content ={'first_name': core_users.first_name, 'last_name': core_users.last_name, 'employee_id':core_users.employees.employee_id,
        'datetime':now, 'distance':dist[0], "state": sc_state}
        return Response(content)

class SetUserTemplateWebapp(APIView):
    permission_classes = (IsOptionsOrIsAuthenticated,)

    def post(self, request):
        try:
            #save temp image 
            dir_path = os.path.join(settings.BASE_DIR, "temp")
            try:
                os.makedirs(dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    #directory already exists
                    pass

            received_json_data=json.loads(request.body)
            image_file = received_json_data['imagen']
            uploaded_file_url = str("IMG_"+str(datetime.now().timestamp()).replace(".","")+"_"+str(datetime.now().year)+str(datetime.now().month)+".jpg")
            path_temp = os.path.join(dir_path,uploaded_file_url)
            img_B64 = image_file.replace("data:image/png;base64,","")
            img_wr = base64.b64decode(img_B64)
            with open(path_temp, 'wb') as image_saved:
                image_saved.write(img_wr)

            #align image and get user template url
            user = User.objects.get(username= request.user.username)
            path_img = aling_image(path_temp_image=path_temp,image_name=uploaded_file_url,username=str(user.employees.id),save_directory="tmplt\\")
                
            core_users = user.employees
            core_users.user_template=path_img
            core_users.save()
            encrypt_image(path_img,core_users.employee_id)

            if os.path.exists(path_temp):
                os.remove(path_temp)

            template = core_users.user_template
            img_B64_Template = "null"  
            if template :
                aux_path = template.url
                img_B64_Template = dencrypt_image(aux_path.replace("%3A", ":"),core_users.employee_id)
            
            content ={'employee_id':core_users.employee_id,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'username' : user.username,
            'email' : user.email,
            'image': img_B64_Template,
            'registration_date' : core_users.registration_data}
            state = status.HTTP_200_OK
        except Exception as e :
            content = {"message" : "Algo inesperado ha ocurrido", "Exception" : str(e)}
            state = status.HTTP_500_INTERNAL_SERVER_ERROR
        finally:
            if os.path.exists(path_temp):
                os.remove(path_temp)

            return Response(content, status= state)