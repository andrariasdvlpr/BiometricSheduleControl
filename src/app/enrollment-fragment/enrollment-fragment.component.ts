import { Component, OnInit } from '@angular/core';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Router } from '@angular/router';
import { LoginService } from '../login.service';
import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { RestService } from '../rest.service';

@Component({
  selector: 'app-enrollment-fragment',
  templateUrl: './enrollment-fragment.component.html',
  styleUrls: ['./enrollment-fragment.component.css']
})
export class EnrollmentFragmentComponent implements OnInit {

  value : any;
  isShowResult : boolean = false;
  dialog!: MatDialogRef<any, any>;
  isInitialState: boolean = true;

  constructor(public restService : RestService, public matDialog : MatDialog,
     public snackBar : MatSnackBar, public router : Router, public loginService : LoginService) { }

  ngOnInit(): void {
    if(!this.loginService.hasToken()){
      this.router.navigateByUrl("/login")
    }else{
      this.restService.getProfileUser().subscribe( data =>{
        if(data.image != "null"){
          this.router.navigateByUrl('/app/home');
        }    
      });
    }
  }

  showResult(data : any){
    this.dialog = this.openProgressDialog();
    this.restService.setUserTemplate(data).subscribe(
      data => {
        this.shapeResponse(data)
      },
      err => {
        this.dialog.close()
        this.openSnackBar(err.statusText, "error-snackbar");
      },
      () =>  this.dialog.close()
    );
    
  }

  shapeResponse(data : any){
    this.value = data
    this.isShowResult = true;
  }

  openProgressDialog() : MatDialogRef<any>{
    const dialogConfig = new MatDialogConfig();
    // The user can't close the dialog by clicking outside its body
    dialogConfig.id = "enroll-modal-component";
    dialogConfig.maxHeight ="90%";
    dialogConfig.disableClose = true;
    const modalDialog= this.matDialog.open(ProgressDialogComponent, dialogConfig);
    return modalDialog;
  }

  openSnackBar(message : string, classStyle : string) {
    this.snackBar.open(message,'', {duration: 5000 ,panelClass: classStyle});
  }

  goHome(){
    this.router.navigateByUrl('/app/home');
  }

  changeInitialState(){
    this.isInitialState = false;
  }

  goLogin(){
    this.router.navigateByUrl('/login');
  }
}
