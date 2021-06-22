import { AfterViewInit, Component, OnInit } from '@angular/core';
import { RestService } from '../rest.service';
import { RecordInfoComponent } from '../record-info/record-info.component';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { Router } from '@angular/router';
import { LoginService } from '../login.service';

@Component({
  selector: 'app-history-fragment',
  templateUrl: './history-fragment.component.html',
  styleUrls: ['./history-fragment.component.css']
})
export class HistoryFragmentComponent implements OnInit {

  data: any[] = [];
  show : boolean = false;
  dialog!: MatDialogRef<any, any>;

  constructor(public restService : RestService,public matDialog: MatDialog,
    public router : Router, public loginService : LoginService) { }

  ngOnInit(): void {
    if(!this.loginService.hasToken()){
      this.router.navigateByUrl("/login")
    }else{
      this.dialog = this.openProgressDialog();
      this.getUserActivityLog();
    }
  }

  getUserActivityLog(){
    this.restService.getActivityLog().subscribe(data => {
      this.data=data;
      this.dialog.close();
    });
  }

  openModal(record : any) {
    this.restService.getActivityLogData(record.sc_id).subscribe(data => {
      console.log(data);
      const dialogConfig = new MatDialogConfig();
      // The user can't close the dialog by clicking outside its body
      dialogConfig.id = "modal-component";
      dialogConfig.maxWidth ="90vw"
      dialogConfig.data = {record : record, ActLog: data}
      // https://material.angular.io/components/dialog/overview
      const modalDialog = this.matDialog.open(RecordInfoComponent, dialogConfig);
    });
  }

  openProgressDialog() : MatDialogRef<any>{
    const dialogConfig = new MatDialogConfig();
    // The user can't close the dialog by clicking outside its body
    dialogConfig.id = "profile-modal-component";
    dialogConfig.maxHeight ="90%";
    dialogConfig.disableClose = true;
    const modalDialog= this.matDialog.open(ProgressDialogComponent, dialogConfig);
    return modalDialog;
  }
}
