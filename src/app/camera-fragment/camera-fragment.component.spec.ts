import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CameraFragmentComponent } from './camera-fragment.component';

describe('CameraFragmentComponent', () => {
  let component: CameraFragmentComponent;
  let fixture: ComponentFixture<CameraFragmentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CameraFragmentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CameraFragmentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
