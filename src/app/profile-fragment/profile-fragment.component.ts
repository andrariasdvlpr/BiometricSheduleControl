import { AfterViewInit, Component, OnInit } from '@angular/core';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { LoginService } from '../login.service';
import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { RestService } from '../rest.service';

@Component({
  selector: 'app-profile-fragment',
  templateUrl: './profile-fragment.component.html',
  styleUrls: ['./profile-fragment.component.css']
})
export class ProfileFragmentComponent implements OnInit {

  userData : any;
  dialog!: MatDialogRef<any, any>;

  constructor(public restService : RestService, public matDialog : MatDialog,
    public router : Router, public loginService : LoginService) { }

  ngOnInit(): void {
    if(!this.loginService.hasToken()){
      this.router.navigateByUrl("/login")
    }else{
      this.dialog = this.openProgressDialog();
      this.gerUserData();
    }
  }

  gerUserData() {
    this.restService.getProfileUser().subscribe( data =>{
      this.userData = data;
      this.dialog.close();
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
