import { LOCALE_ID, NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { CookieService } from 'ngx-cookie-service';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { LateralNavBarComponent } from './lateral-nav-bar/lateral-nav-bar.component';
import { TopBarComponent } from './top-bar/top-bar.component';
import { HistoryComponent } from './history/history.component';
import { RecordInfoComponent } from './record-info/record-info.component';
import { LoginComponent } from './login/login.component';
import { HistoryFragmentComponent } from './history-fragment/history-fragment.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatDialogModule } from '@angular/material/dialog';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import localeEs from '@angular/common/locales/es';
import { registerLocaleData } from '@angular/common';
import { MainContainerComponent } from './main-container/main-container.component';
import { HomeFragmentComponent } from './home-fragment/home-fragment.component';
import { ProfileFragmentComponent } from './profile-fragment/profile-fragment.component';
import { CameraFragmentComponent } from './camera-fragment/camera-fragment.component';
import { PasswordFragmentComponent } from './password-fragment/password-fragment.component';
import { MapComponent } from './map/map.component';
import { CameraComponent } from './camera/camera.component';
import { ProgressDialogComponent } from './progress-dialog/progress-dialog.component';
import { RegisterFragmentComponent } from './register-fragment/register-fragment.component';
import { EnrollmentFragmentComponent } from './enrollment-fragment/enrollment-fragment.component';
import {MatSelectModule} from '@angular/material/select';

registerLocaleData(localeEs, 'es');

@NgModule({
  declarations: [
    AppComponent,
    LateralNavBarComponent,
    TopBarComponent,
    HistoryComponent,
    RecordInfoComponent,
    LoginComponent,
    HistoryFragmentComponent,
    MainContainerComponent,
    HomeFragmentComponent,
    ProfileFragmentComponent,
    CameraFragmentComponent,
    PasswordFragmentComponent,
    MapComponent,
    CameraComponent,
    ProgressDialogComponent,
    RegisterFragmentComponent,
    EnrollmentFragmentComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    BrowserAnimationsModule,
    MatDialogModule,
    MatToolbarModule,
    MatIconModule,
    MatSnackBarModule,
    MatProgressSpinnerModule,
    MatSelectModule
  ],
  providers: [CookieService,{ provide: LOCALE_ID, useValue: 'es' }],
  bootstrap: [AppComponent],
  entryComponents : [RecordInfoComponent,MapComponent, ProgressDialogComponent]
})
export class AppModule { }
