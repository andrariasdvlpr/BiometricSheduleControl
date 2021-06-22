import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnrollmentFragmentComponent } from './enrollment-fragment.component';

describe('EnrollmentFragmentComponent', () => {
  let component: EnrollmentFragmentComponent;
  let fixture: ComponentFixture<EnrollmentFragmentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ EnrollmentFragmentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(EnrollmentFragmentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
