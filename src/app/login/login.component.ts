import { Component, OnInit, Input, AfterViewInit } from '@angular/core';
import { LoginService } from '../login.service';
import { Router} from '@angular/router';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { RestService } from '../rest.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit, AfterViewInit {

  username!: string;
  password!: string;

  constructor(public loginService : LoginService, public router : Router, public snackBar : MatSnackBar,
    public matDialog : MatDialog, public restService : RestService) {
   }

  ngOnInit(): void {
    if(this.loginService.hasToken()){
      this.loginService.deleteToken();
    }
  }

  ngAfterViewInit(){
    if(this.loginService.hasToken()){
      this.loginService.deleteToken();
    }
  }

  login(){
    if(this.username && this.password){
      const user ={ username : this.username, password : this.password};
      const dialog = this.openProgressDialog();
      this.loginService.login(user).subscribe(
        res => {
          this.loginService.setToken(res.token);
          this.restService.getProfileUser().subscribe( data =>{
            dialog.close();
            if(data.image == "null"){
              this.router.navigateByUrl('/biometric/enroll');
            }else{
              this.router.navigateByUrl('/app/home');
            }       
          });
          
      }, err => {
        dialog.close();
        if(err.status == 400){
          this.openSnackBar("Usuario y/o contraseña incorrectas");
        }
      });
    }else{
      this.openSnackBar("No deje campos vacios");
    }

  }

  goRegisterFragment(){
    this.router.navigateByUrl('/enroll');
  }

  openSnackBar(message : string) {
    this.snackBar.open(message,'', {duration: 5000 ,panelClass:"error-snackbar"});
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
