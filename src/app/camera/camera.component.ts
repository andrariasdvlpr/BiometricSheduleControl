import { Component, OnInit, Output, EventEmitter} from '@angular/core';
import { ViewChild } from '@angular/core';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { RestService } from '../rest.service';
declare var faceapi: any;



@Component({
  selector: 'app-camera',
  templateUrl: './camera.component.html',
  styleUrls: ['./camera.component.css']
})
export class CameraComponent implements OnInit {

  @ViewChild('videoElement') videoElement: any;
  @ViewChild('canvasFace') canvasElement : any;
  video: any;
  canvas : any;

  @Output() resultEvent = new EventEmitter<any>();
  dialog!: MatDialogRef<any, any>;

  ngOnInit() {

  }

  ngAfterViewInit(){
    this.dialog = this.openProgressDialog();
    this.video = this.videoElement.nativeElement;
    this.canvas = this.canvasElement.nativeElement;
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('../../assets/models'),
      //faceapi.nets.faceRecognitionNet.loadFromUri('../../assets/models')
    ]);
    this.initCamera({ audio: false, video: { width: 640, height: 480 } });
  }

  constructor(public restService : RestService, public matDialog : MatDialog) { }

  initCamera(config:any) {
    navigator.mediaDevices.getUserMedia(config).then((stream: any) => {
      this.video.srcObject =stream;
      this.video.addEventListener('play', () => {
        this.dialog.close();
        const displaySize = { width: this.video.width, height: this.video.height }
        faceapi.matchDimensions(this.canvas, displaySize)

        var interval = setInterval(async () => {
          const detections = await faceapi.detectAllFaces(this.video, new faceapi.TinyFaceDetectorOptions())

          if(detections.length == 1){
            this.canvas.style.display ="none";
            this.video.style.display ="none";
            this.video.pause();
            this.canvas.getContext('2d').drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            var data = this.canvas.toDataURL();
            console.log(data);
            this.video.currentTime =0
            clearInterval(interval);
            this.stopVideoOnly(this.video.srcObject);
            if(navigator.geolocation){
              navigator.geolocation.getCurrentPosition( position => {
              const locationFc = { latitude : position.coords.latitude, longitude : position.coords.longitude};
              const bodyPost = { imagen : data.replace("data:image/png;base64,",""), location : locationFc }
              console.log(bodyPost);
              this.resultEvent.emit(bodyPost);
                 
              });
            }
          }
          //console.log(detections)
        }, 3000)
      });
    });

  }

  stopVideoOnly(stream : any) {
    stream.getTracks().forEach( (track: { readyState: string; kind: string; stop: () => void; }) => {
        if (track.readyState == 'live' && track.kind === 'video') {
            track.stop();
        }
    });
}

  openProgressDialog() : MatDialogRef<any>{
    const dialogConfig = new MatDialogConfig();
    // The user can't close the dialog by clicking outside its body
    dialogConfig.id = "map-modal-component";
    dialogConfig.maxHeight ="90%";
    dialogConfig.disableClose = true;
    const modalDialog= this.matDialog.open(ProgressDialogComponent, dialogConfig);
    return modalDialog;
  }
}
