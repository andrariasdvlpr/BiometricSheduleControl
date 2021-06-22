import { Component, OnInit, Output, EventEmitter, Inject } from '@angular/core';
import {MatDialog, MatDialogConfig, MAT_DIALOG_DATA} from '@angular/material/dialog';
import { MapComponent } from '../map/map.component';

@Component({
  selector: 'app-record-info',
  templateUrl: './record-info.component.html',
  styleUrls: ['./record-info.component.css']
})
export class RecordInfoComponent implements OnInit {

  @Output() closeEvent = new EventEmitter<boolean>();
  record : any;
  ActLog : any;

  constructor(@Inject(MAT_DIALOG_DATA) public data :{record: any , ActLog : any} , public matDialog : MatDialog) {
    this.record =data.record;
    this.ActLog = data.ActLog;
  }

  ngOnInit(): void {
  }

  closeModal(value: boolean) {
    this.closeEvent.emit(!value);
  }

  showLocation(){
    const dialogConfig = new MatDialogConfig();
      // The user can't close the dialog by clicking outside its body
      dialogConfig.id = "map-modal-component";
      dialogConfig.maxHeight ="90%";
      dialogConfig.data = {longitude : this.ActLog.location.longitude, latitude: this.ActLog.location.latitude}
      // https://material.angular.io/components/dialog/overview
      const modalDialog = this.matDialog.open(MapComponent, dialogConfig);
  }

}
