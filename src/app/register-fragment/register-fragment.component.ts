import { Component, OnInit } from '@angular/core';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Router } from '@angular/router';

import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { RestService } from '../rest.service';

@Component({
  selector: 'app-register-fragment',
  templateUrl: './register-fragment.component.html',
  styleUrls: ['./register-fragment.component.css']
})
export class RegisterFragmentComponent implements OnInit {

  username!: string;
  password!: string;
  email!: string;
  lastname!: string;
  firstname!: string;
  existUsername : boolean = false;
  existEmail : boolean = false;
  wait: boolean = false;

  constructor(public restService : RestService, public router : Router, public snackBar : MatSnackBar,
    public matDialog : MatDialog) { }

  ngOnInit(): void {
  }

  enroll(){
    if(!this.wait){
      if(this.username && this.password && this.email && this.firstname && this.lastname
        && !this.existEmail && !this.existUsername){
        const user ={ 
          firstname : this.firstname,
          lastname : this.lastname,
          username : this.username, 
          password : this.password,
          email : this.email,
        };
        const dialog = this.openProgressDialog();
        this.restService.registerUser(user).subscribe(
          res => {
            if(res.result.localeCompare("ok") == 0){
              dialog.close();
              this.openSnackBar("Usuario creado","success-snackbar")
              this.router.navigateByUrl('/login');
            }else{
              this.openSnackBar(res.message,"error-snackbar");
            }
        }, err => {
          dialog.close();
            this.openSnackBar("Error","error-snackbar");
        });
      }else{
        if(this.existEmail){
          this.openSnackBar("El correo ya existe","error-snackbar");
        }
        else if(this.existUsername){
          this.openSnackBar("El usuario ya existe","error-snackbar");
        }else{
          this.openSnackBar("No deje campos vacios","error-snackbar");
        }
      }
    }else{
      this.openSnackBar("Comprobando los datos, espere","warning-snackbar");
    }
  }

  isUsedUser(){
    const data = {username : this.username}
    this.wait = true;
    this.restService.isUsedEmailOrUser(data).subscribe(
      res => {
        if(res.result.localeCompare("exist") == 0){
          this.openSnackBar("El usuario ya existe","error-snackbar");
          this.existUsername = true;
        }else{
          this.existUsername = false;
        }
      }, err => {
          this.openSnackBar("Error","error-snackbar");
      },
      () => this.wait = false);
  }

  isUsedEmail(){
    const data = {email : this.email}
    this.wait = true;
    this.restService.isUsedEmailOrUser(data).subscribe(
      res => {
        if(res.result.localeCompare("exist") == 0){
          this.openSnackBar("El correo ya existe","error-snackbar");
          this.existEmail = true;
        }else{
          this.existEmail = false;
        }
      }, err => {
          this.openSnackBar("Error", "error-snackbar");
      },
      () => this.wait = false);
  }

  openSnackBar(message : string, classStyle : string) {
    this.snackBar.open(message,'', {duration: 5000 ,panelClass: classStyle});
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
